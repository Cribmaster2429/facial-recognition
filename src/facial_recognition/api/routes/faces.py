"""Face registration and management endpoints."""

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from facial_recognition.api.dependencies import get_recognizer, get_storage
from facial_recognition.api.schemas import (
    ErrorResponse,
    PersonResponse,
    PersonUpdate,
)
from facial_recognition.core.recognizer import FaceRecognizer
from facial_recognition.storage.base import BaseStorage
from facial_recognition.utils.image import load_image_from_upload

router = APIRouter(prefix="/faces", tags=["Face Management"])


@router.post(
    "/register",
    response_model=PersonResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid image or no face detected"},
    },
    summary="Register a new person",
    description="Register a new person by uploading an image containing their face.",
)
async def register_face(
    name: Annotated[str, Form(description="Person's name")],
    image: Annotated[UploadFile, File(description="Image file containing the face")],
    recognizer: FaceRecognizer = Depends(get_recognizer),
    storage: BaseStorage = Depends(get_storage),
) -> PersonResponse:
    """Register a new person with their face image."""
    try:
        img = await load_image_from_upload(image)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        person = recognizer.register_person(name=name, image=img)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    await storage.save_person(person)

    return PersonResponse(
        id=person.id,
        name=person.name,
        metadata=person.metadata,
        created_at=person.created_at,
        updated_at=person.updated_at,
    )


@router.get(
    "",
    response_model=list[PersonResponse],
    summary="List all registered persons",
    description="Get a list of all registered persons.",
)
async def list_persons(
    recognizer: FaceRecognizer = Depends(get_recognizer),
) -> list[PersonResponse]:
    """List all registered persons."""
    return [
        PersonResponse(
            id=p.id,
            name=p.name,
            metadata=p.metadata,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
        for p in recognizer.known_persons
    ]


@router.get(
    "/{person_id}",
    response_model=PersonResponse,
    responses={404: {"model": ErrorResponse, "description": "Person not found"}},
    summary="Get person by ID",
    description="Retrieve details of a registered person by their ID.",
)
async def get_person(
    person_id: UUID,
    recognizer: FaceRecognizer = Depends(get_recognizer),
) -> PersonResponse:
    """Get a person by their ID."""
    person = recognizer.get_person(str(person_id))
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    return PersonResponse(
        id=person.id,
        name=person.name,
        metadata=person.metadata,
        created_at=person.created_at,
        updated_at=person.updated_at,
    )


@router.patch(
    "/{person_id}",
    response_model=PersonResponse,
    responses={404: {"model": ErrorResponse, "description": "Person not found"}},
    summary="Update person",
    description="Update a registered person's details.",
)
async def update_person(
    person_id: UUID,
    update: PersonUpdate,
    recognizer: FaceRecognizer = Depends(get_recognizer),
    storage: BaseStorage = Depends(get_storage),
) -> PersonResponse:
    """Update a person's details."""
    person = recognizer.get_person(str(person_id))
    if person is None:
        raise HTTPException(status_code=404, detail="Person not found")

    if update.name is not None:
        person.name = update.name
    if update.metadata is not None:
        person.metadata = update.metadata

    await storage.update_person(person)

    return PersonResponse(
        id=person.id,
        name=person.name,
        metadata=person.metadata,
        created_at=person.created_at,
        updated_at=person.updated_at,
    )


@router.delete(
    "/{person_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse, "description": "Person not found"}},
    summary="Delete person",
    description="Remove a registered person from the system.",
)
async def delete_person(
    person_id: UUID,
    recognizer: FaceRecognizer = Depends(get_recognizer),
    storage: BaseStorage = Depends(get_storage),
) -> None:
    """Delete a registered person."""
    success = recognizer.unregister_person(str(person_id))
    if not success:
        raise HTTPException(status_code=404, detail="Person not found")

    await storage.delete_person(str(person_id))


@router.get(
    "/search/{name}",
    response_model=list[PersonResponse],
    summary="Search persons by name",
    description="Search for persons by name (case-insensitive partial match).",
)
async def search_persons(
    name: str,
    recognizer: FaceRecognizer = Depends(get_recognizer),
) -> list[PersonResponse]:
    """Search persons by name."""
    persons = recognizer.find_person_by_name(name)
    return [
        PersonResponse(
            id=p.id,
            name=p.name,
            metadata=p.metadata,
            created_at=p.created_at,
            updated_at=p.updated_at,
        )
        for p in persons
    ]
