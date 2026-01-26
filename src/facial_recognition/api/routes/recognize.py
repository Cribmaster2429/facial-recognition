"""Face recognition and detection endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from facial_recognition.api.dependencies import get_detector, get_recognizer
from facial_recognition.api.schemas import (
    BoundingBoxResponse,
    DetectedFaceResponse,
    DetectResponse,
    ErrorResponse,
    MatchResponse,
    PersonResponse,
    RecognitionResultResponse,
    RecognizeResponse,
    VerifyResponse,
)
from facial_recognition.core.detector import FaceDetector
from facial_recognition.core.recognizer import FaceRecognizer
from facial_recognition.utils.image import load_image_from_upload

router = APIRouter(prefix="/recognize", tags=["Recognition"])


@router.post(
    "",
    response_model=RecognizeResponse,
    responses={400: {"model": ErrorResponse, "description": "Invalid image"}},
    summary="Recognize faces in image",
    description="Detect and recognize all faces in an uploaded image.",
)
async def recognize_faces(
    image: Annotated[UploadFile, File(description="Image file to analyze")],
    recognizer: FaceRecognizer = Depends(get_recognizer),
) -> RecognizeResponse:
    """Recognize faces in an uploaded image."""
    try:
        img = await load_image_from_upload(image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    results = recognizer.recognize(img)

    response_results = []
    faces_recognized = 0

    for result in results:
        bbox = result.face.bounding_box
        face_response = DetectedFaceResponse(
            bounding_box=BoundingBoxResponse(
                top=bbox.top,
                right=bbox.right,
                bottom=bbox.bottom,
                left=bbox.left,
                width=bbox.width,
                height=bbox.height,
            ),
            confidence=result.face.confidence,
        )

        match_response = None
        if result.match:
            faces_recognized += 1
            match_response = MatchResponse(
                person=PersonResponse(
                    id=result.match.person.id,
                    name=result.match.person.name,
                    metadata=result.match.person.metadata,
                    created_at=result.match.person.created_at,
                    updated_at=result.match.person.updated_at,
                ),
                distance=result.match.distance,
                confidence=result.match.confidence,
            )

        response_results.append(
            RecognitionResultResponse(
                face=face_response,
                match=match_response,
                recognized=result.recognized,
            )
        )

    return RecognizeResponse(
        faces_detected=len(results),
        faces_recognized=faces_recognized,
        results=response_results,
    )


@router.post(
    "/detect",
    response_model=DetectResponse,
    responses={400: {"model": ErrorResponse, "description": "Invalid image"}},
    summary="Detect faces in image",
    description="Detect all faces in an uploaded image without recognition.",
)
async def detect_faces(
    image: Annotated[UploadFile, File(description="Image file to analyze")],
    detector: FaceDetector = Depends(get_detector),
) -> DetectResponse:
    """Detect faces in an uploaded image."""
    try:
        img = await load_image_from_upload(image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    faces = detector.detect(img)

    face_responses = []
    for face in faces:
        bbox = face.bounding_box
        face_responses.append(
            DetectedFaceResponse(
                bounding_box=BoundingBoxResponse(
                    top=bbox.top,
                    right=bbox.right,
                    bottom=bbox.bottom,
                    left=bbox.left,
                    width=bbox.width,
                    height=bbox.height,
                ),
                confidence=face.confidence,
            )
        )

    return DetectResponse(
        faces_detected=len(faces),
        faces=face_responses,
    )


@router.post(
    "/verify/{person_id}",
    response_model=VerifyResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image or no face"},
        404: {"model": ErrorResponse, "description": "Person not found"},
    },
    summary="Verify face against person",
    description="Verify if an image contains a specific registered person.",
)
async def verify_face(
    person_id: UUID,
    image: Annotated[UploadFile, File(description="Image file to verify")],
    recognizer: FaceRecognizer = Depends(get_recognizer),
) -> VerifyResponse:
    """Verify a face against a registered person."""
    try:
        img = await load_image_from_upload(image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        verified, confidence = recognizer.verify(img, str(person_id))
    except ValueError as e:
        error_msg = str(e)
        if "not found" in error_msg.lower():
            raise HTTPException(status_code=404, detail=error_msg)
        raise HTTPException(status_code=400, detail=error_msg)

    return VerifyResponse(
        verified=verified,
        confidence=confidence,
        person_id=person_id,
    )
