"""CLI commands using Typer."""

import asyncio
from pathlib import Path
from typing import Optional

import cv2
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from facial_recognition import __version__
from facial_recognition.core.detector import FaceDetector
from facial_recognition.core.encoder import FaceEncoder
from facial_recognition.core.recognizer import FaceRecognizer
from facial_recognition.storage.sqlite import SQLiteStorage
from facial_recognition.utils.image import draw_face_boxes, load_image_from_path, save_image

app = typer.Typer(
    name="facial-recognition",
    help="Real-time facial recognition system using deep learning.",
    add_completion=False,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"Facial Recognition v{__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Facial Recognition CLI - Detect, register, and recognize faces."""
    pass


@app.command()
def detect(
    image: Path = typer.Argument(..., help="Path to image file", exists=True),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save annotated image to this path"
    ),
    model: str = typer.Option("hog", "--model", "-m", help="Detection model: hog or cnn"),
) -> None:
    """Detect faces in an image."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Detecting faces...", total=None)

        detector = FaceDetector(model=model)  # type: ignore
        img = load_image_from_path(image)
        faces = detector.detect(img)

    if not faces:
        console.print("[yellow]No faces detected in the image.[/yellow]")
        raise typer.Exit(1)

    console.print(f"[green]Detected {len(faces)} face(s)[/green]")

    table = Table(title="Detected Faces")
    table.add_column("Face #", style="cyan")
    table.add_column("Position (top, right, bottom, left)")
    table.add_column("Size (WxH)")

    for i, face in enumerate(faces, 1):
        bbox = face.bounding_box
        table.add_row(
            str(i),
            f"({bbox.top}, {bbox.right}, {bbox.bottom}, {bbox.left})",
            f"{bbox.width}x{bbox.height}",
        )

    console.print(table)

    if output:
        boxes = [f.bounding_box.to_tuple() for f in faces]
        annotated = draw_face_boxes(img, boxes)
        save_image(annotated, output)
        console.print(f"[green]Saved annotated image to {output}[/green]")


@app.command()
def register(
    name: str = typer.Argument(..., help="Name of the person to register"),
    image: Path = typer.Argument(..., help="Path to image file", exists=True),
    database: Path = typer.Option(
        "./data/faces.db", "--database", "-d", help="Path to database file"
    ),
) -> None:
    """Register a person from an image."""

    async def _register() -> None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Processing...", total=None)

            recognizer = FaceRecognizer()
            img = load_image_from_path(image)

            progress.update(task, description="Detecting and encoding face...")
            try:
                person = recognizer.register_person(name=name, image=img)
            except ValueError as e:
                console.print(f"[red]Error: {e}[/red]")
                raise typer.Exit(1)

            progress.update(task, description="Saving to database...")
            async with SQLiteStorage(database) as storage:
                await storage.save_person(person)

        console.print(f"[green]Successfully registered '{name}'[/green]")
        console.print(f"Person ID: [cyan]{person.id}[/cyan]")

    asyncio.run(_register())


@app.command()
def recognize(
    image: Path = typer.Argument(..., help="Path to image file", exists=True),
    database: Path = typer.Option(
        "./data/faces.db", "--database", "-d", help="Path to database file"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Save annotated image to this path"
    ),
    tolerance: float = typer.Option(
        0.6, "--tolerance", "-t", help="Matching tolerance (lower = stricter)"
    ),
) -> None:
    """Recognize faces in an image."""

    async def _recognize() -> None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Loading database...", total=None)

            recognizer = FaceRecognizer(tolerance=tolerance)

            async with SQLiteStorage(database) as storage:
                persons = await storage.get_all_persons()
                recognizer.load_persons(persons)

            if not persons:
                console.print("[yellow]No registered persons in database.[/yellow]")
                raise typer.Exit(1)

            progress.update(task, description=f"Loaded {len(persons)} person(s)")

            img = load_image_from_path(image)
            progress.update(task, description="Recognizing faces...")
            results = recognizer.recognize(img)

        if not results:
            console.print("[yellow]No faces detected in the image.[/yellow]")
            raise typer.Exit(1)

        recognized_count = sum(1 for r in results if r.recognized)
        console.print(
            f"[green]Found {len(results)} face(s), "
            f"recognized {recognized_count}[/green]"
        )

        table = Table(title="Recognition Results")
        table.add_column("Face #", style="cyan")
        table.add_column("Name")
        table.add_column("Confidence")
        table.add_column("Status")

        names = []
        boxes = []

        for i, result in enumerate(results, 1):
            boxes.append(result.face.bounding_box.to_tuple())

            if result.match:
                name = result.match.person.name
                confidence = f"{result.match.confidence:.1%}"
                status = "[green]Recognized[/green]"
                names.append(f"{name} ({confidence})")
            else:
                name = "Unknown"
                confidence = "-"
                status = "[yellow]Unknown[/yellow]"
                names.append("Unknown")

            table.add_row(str(i), name, confidence, status)

        console.print(table)

        if output:
            annotated = draw_face_boxes(img, boxes, names)
            save_image(annotated, output)
            console.print(f"[green]Saved annotated image to {output}[/green]")

    asyncio.run(_recognize())


@app.command("list")
def list_persons(
    database: Path = typer.Option(
        "./data/faces.db", "--database", "-d", help="Path to database file"
    ),
) -> None:
    """List all registered persons."""

    async def _list() -> None:
        async with SQLiteStorage(database) as storage:
            persons = await storage.get_all_persons()

        if not persons:
            console.print("[yellow]No registered persons.[/yellow]")
            raise typer.Exit(0)

        table = Table(title=f"Registered Persons ({len(persons)})")
        table.add_column("ID", style="cyan")
        table.add_column("Name")
        table.add_column("Created")

        for person in persons:
            table.add_row(
                str(person.id)[:8] + "...",
                person.name,
                person.created_at.strftime("%Y-%m-%d %H:%M"),
            )

        console.print(table)

    asyncio.run(_list())


@app.command()
def delete(
    person_id: str = typer.Argument(..., help="ID of the person to delete"),
    database: Path = typer.Option(
        "./data/faces.db", "--database", "-d", help="Path to database file"
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
) -> None:
    """Delete a registered person."""

    async def _delete() -> None:
        async with SQLiteStorage(database) as storage:
            person = await storage.get_person(person_id)

            if not person:
                console.print(f"[red]Person not found: {person_id}[/red]")
                raise typer.Exit(1)

            if not force:
                confirm = typer.confirm(f"Delete '{person.name}'?")
                if not confirm:
                    console.print("[yellow]Cancelled.[/yellow]")
                    raise typer.Exit(0)

            await storage.delete_person(person_id)

        console.print(f"[green]Deleted '{person.name}'[/green]")

    asyncio.run(_delete())


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload"),
) -> None:
    """Start the REST API server."""
    import uvicorn

    console.print(f"[green]Starting API server at http://{host}:{port}[/green]")
    console.print("[dim]Press Ctrl+C to stop[/dim]")

    uvicorn.run(
        "facial_recognition.api.app:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def webcam(
    database: Path = typer.Option(
        "./data/faces.db", "--database", "-d", help="Path to database file"
    ),
    tolerance: float = typer.Option(
        0.6, "--tolerance", "-t", help="Matching tolerance"
    ),
    scale: float = typer.Option(
        0.25, "--scale", "-s", help="Frame scale for processing (lower = faster)"
    ),
) -> None:
    """Run real-time face recognition from webcam."""

    async def _load_persons() -> list:
        async with SQLiteStorage(database) as storage:
            return await storage.get_all_persons()

    persons = asyncio.run(_load_persons())

    recognizer = FaceRecognizer(tolerance=tolerance)
    recognizer.load_persons(persons)
    detector = FaceDetector()
    encoder = FaceEncoder()

    console.print(f"[green]Loaded {len(persons)} registered person(s)[/green]")
    console.print("[dim]Press 'q' to quit[/dim]")

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        console.print("[red]Error: Could not open webcam[/red]")
        raise typer.Exit(1)

    process_frame = True

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if process_frame:
                faces = detector.detect_from_video_frame(frame, scale=scale)
                encoder.encode_from_video_frame(frame, faces)

                names = []
                boxes = []

                for face in faces:
                    boxes.append(face.bounding_box.to_tuple())

                    if face.encoding:
                        match = recognizer.recognize_face(face.encoding)
                        if match:
                            names.append(f"{match.person.name} ({match.confidence:.0%})")
                        else:
                            names.append("Unknown")
                    else:
                        names.append("Unknown")

                for (top, right, bottom, left), name in zip(boxes, names):
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

                    cv2.rectangle(
                        frame, (left, bottom - 25), (right, bottom), color, cv2.FILLED
                    )
                    cv2.putText(
                        frame,
                        name,
                        (left + 6, bottom - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1,
                    )

            process_frame = not process_frame

            cv2.imshow("Facial Recognition", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app()
