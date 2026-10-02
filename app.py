# app.py
# GBSBFORYOU Voice API v1
# Hindi Voice Cloning API using XTTS-v2

import os
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from TTS.api import TTS


# --------------------------------------------------
# APP
# --------------------------------------------------

app = FastAPI(
    title="GBSBFORYOU Voice API",
    description="Hindi Voice Cloning API powered by XTTS-v2",
    version="1.0.0",
)


# --------------------------------------------------
# CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"

SUPPORTED_LANGUAGES = {
    "hi": "Hindi",
    "en": "English",
}

MAX_TEXT_LENGTH = 5000
MAX_VOICE_SIZE = 20 * 1024 * 1024  # 20 MB


# --------------------------------------------------
# MODEL
# --------------------------------------------------

tts_model = None


def get_model():
    global tts_model

    if tts_model is None:
        use_gpu = os.getenv("USE_GPU", "false").lower() == "true"

        tts_model = TTS(
            model_name=MODEL_NAME,
            progress_bar=False,
            gpu=use_gpu,
        )

    return tts_model


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "service": "GBSBFORYOU Voice API",
        "version": "1.0.0",
        "status": "online",
        "engine": "XTTS-v2",
        "languages": SUPPORTED_LANGUAGES,
        "endpoints": {
            "health": "/health",
            "clone": "/clone",
        },
    }


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "GBSBFORYOU Voice API",
        "version": "1.0.0",
        "engine": "XTTS-v2",
        "model_loaded": tts_model is not None,
        "languages": SUPPORTED_LANGUAGES,
    }


# --------------------------------------------------
# VOICE CLONE
# --------------------------------------------------

@app.post("/clone")
async def clone_voice(
    voice: UploadFile = File(...),
    text: str = Form(...),
    language: str = Form("hi"),
    voice_name: str = Form(
        "Ajay Singh Chouhan — Professor Narrator"
    ),
):
    """
    Generate speech in the reference speaker's voice.

    Parameters:
    - voice: reference voice WAV/MP3/M4A/OGG
    - text: text to speak
    - language: hi or en
    - voice_name: profile name
    """

    # ----------------------------------------------
    # Validate language
    # ----------------------------------------------

    language = language.lower().strip()

    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Unsupported language",
                "supported_languages": SUPPORTED_LANGUAGES,
            },
        )

    # ----------------------------------------------
    # Validate text
    # ----------------------------------------------

    text = text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Text is required.",
        )

    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=f"Text cannot exceed {MAX_TEXT_LENGTH} characters.",
        )

    # ----------------------------------------------
    # Validate voice file
    # ----------------------------------------------

    if not voice.filename:
        raise HTTPException(
            status_code=400,
            detail="Reference voice file is required.",
        )

    allowed_extensions = {
        ".wav",
        ".mp3",
        ".m4a",
        ".ogg",
        ".flac",
    }

    extension = Path(voice.filename).suffix.lower()

    if extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format.",
        )

    # ----------------------------------------------
    # Save reference voice temporarily
    # ----------------------------------------------

    reference_path = None
    output_path = None

    try:
        reference_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        )

        reference_path = reference_file.name

        total_size = 0

        while True:
            chunk = await voice.read(1024 * 1024)

            if not chunk:
                break

            total_size += len(chunk)

            if total_size > MAX_VOICE_SIZE:
                reference_file.close()
                os.unlink(reference_path)

                raise HTTPException(
                    status_code=413,
                    detail="Reference voice file is too large. Maximum size is 20 MB.",
                )

            reference_file.write(chunk)

        reference_file.close()

        # ------------------------------------------
        # Output WAV
        # ------------------------------------------

        output_file = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".wav",
        )

        output_path = output_file.name
        output_file.close()

        # ------------------------------------------
        # Load XTTS-v2
        # ------------------------------------------

        model = get_model()

        # ------------------------------------------
        # Generate cloned voice
        # ------------------------------------------

        model.tts_to_file(
            text=text,
            speaker_wav=reference_path,
            language=language,
            file_path=output_path,
        )

        # ------------------------------------------
        # Return generated audio
        # ------------------------------------------

        response = FileResponse(
            output_path,
            media_type="audio/wav",
            filename="gbsbforyou-voice.wav",
        )

        response.headers["X-Voice-Engine"] = "XTTS-v2"
        response.headers["X-Voice-Name"] = voice_name
        response.headers["X-Language"] = language

        return response

    except HTTPException:
        raise

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "ok": False,
                "error": "Voice generation failed.",
                "message": str(e),
            },
        )

    finally:
        # ------------------------------------------
        # Cleanup reference file
        # ------------------------------------------

        if reference_path:
            try:
                if os.path.exists(reference_path):
                    os.unlink(reference_path)
            except Exception:
                pass

        # Output file is intentionally kept long enough
        # for FileResponse to serve it.
        #
        # A later production version will use a proper
        # background cleanup mechanism.


# --------------------------------------------------
# SERVER
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
)
