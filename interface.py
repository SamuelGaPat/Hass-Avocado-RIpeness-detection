

import asyncio
import base64
import platform

import cv2
import flet as ft

import detection


# =============================================================
# CONFIGURATION
# =============================================================

CAMERA_INDEX = 0

# Reduced camera resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Fixed size for information boxes
INFO_BOX_WIDTH = 300
INFO_BOX_HEIGHT = 220


# =============================================================
# PLACEHOLDER IMAGE
# =============================================================

PLACEHOLDER = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
)


# =============================================================
# STATUS COLORS
# =============================================================

STATUS_COLORS = {
    "Breaking": ft.Colors.AMBER_700,
    "Ripe_first_stage": ft.Colors.LIGHT_GREEN_700,
    "Ripe_second_stage": ft.Colors.GREEN_700,
    "Underripe": ft.Colors.BLUE_700,
    "Overripe": ft.Colors.RED_700,
}


# =============================================================
# CAMERA WORKER
# =============================================================

class CameraWorker:
    """Reads frames and performs classification in a worker thread."""

    def __init__(self, on_result, on_error):
        self.on_result = on_result
        self.on_error = on_error

        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def start(self, index: int):
        if self.running:
            return

        self._stop = asyncio.Event()
        self._task = asyncio.create_task(
            self._loop(index)
        )

    async def stop(self):

        self._stop.set()

        if self._task:
            await self._task

        self._task = None

    def _grab_and_classify(self, cap):
        """
        Runs in a worker thread.

        Captures a reduced-resolution frame and sends the
        raw OpenCV frame to detection.classify().
        """

        ok, raw_frame = cap.read()

        if not ok:
            return None

        # ---------------------------------------------------------
        # CLASSIFICATION
        # ---------------------------------------------------------
        detector = detection.Avocado_detector()
        frame, top1_name, top1_conf = detector.classify(
            raw_frame
        )

        # ---------------------------------------------------------
        # ENCODE FRAME FOR FLET
        # ---------------------------------------------------------

        ok, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80],
        )

        if not ok:
            return None

        return (
            buffer.tobytes(),
            top1_name,
            top1_conf,
        )

    async def _loop(self, index: int):

        # DirectShow on Windows
        backend = (
            cv2.CAP_DSHOW
            if platform.system() == "Windows"
            else cv2.CAP_ANY
        )

        cap = await asyncio.to_thread(
            cv2.VideoCapture,
            index,
            backend,
        )

        try:

            if not cap.isOpened():

                await self.on_error(
                    f"Could not open camera {index}."
                )

                return

            # =====================================================
            # REDUCE CAMERA RESOLUTION
            # =====================================================

            cap.set(
                cv2.CAP_PROP_FRAME_WIDTH,
                CAMERA_WIDTH,
            )

            cap.set(
                cv2.CAP_PROP_FRAME_HEIGHT,
                CAMERA_HEIGHT,
            )

            # =====================================================
            # CAMERA LOOP
            # =====================================================

            while not self._stop.is_set():

                result = await asyncio.to_thread(
                    self._grab_and_classify,
                    cap,
                )

                if result is None:

                    await self.on_error(
                        "Camera stopped returning frames."
                    )

                    break

                jpeg_data, top1_name, top1_conf = result

                self.on_result(
                    jpeg_data,
                    top1_name,
                    top1_conf,
                )

        finally:

            await asyncio.to_thread(
                cap.release
            )


# =============================================================
# MAIN FLET APPLICATION
# =============================================================

async def main(page: ft.Page):

    # =========================================================
    # PAGE CONFIGURATION
    # =========================================================

    page.title = "Hass Avocado Ripeness Classification"

    page.theme_mode = ft.ThemeMode.DARK

    # Remove all margins
    page.padding = 0

    # ---------------------------------------------------------
    # TRUE FULLSCREEN
    # ---------------------------------------------------------

    page.window.full_screen = True

    page.window.resizable = False

    # =========================================================
    # TITLE
    # =========================================================

    title = ft.Text(
        "Hass Avocado Ripeness Classification",
        size=30,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.WHITE,
        text_align=ft.TextAlign.CENTER,
    )

    # =========================================================
    # VIDEO
    # =========================================================

    video = ft.Image(
        src=PLACEHOLDER,

        # Keep camera aspect ratio
        fit=ft.BoxFit.CONTAIN,

        # Prevent flickering
        gapless_playback=True,

        expand=True,
    )

    video_container = ft.Container(
        content=video,

        expand=True,

        bgcolor=ft.Colors.BLACK,

        alignment=ft.Alignment.CENTER,

        border_radius=10,
    )

    # =========================================================
    # AVOCADO STATUS
    # =========================================================

    status_title = ft.Text(
        "Avocado status",
        size=22,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.WHITE,
        text_align=ft.TextAlign.CENTER,
    )

    status_value = ft.Text(
        "Waiting...",
        size=28,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.WHITE,
        text_align=ft.TextAlign.CENTER,
    )

    status_container = ft.Container(
        content=ft.Column(
            [
                status_title,
                status_value,
            ],

            horizontal_alignment=(
                ft.CrossAxisAlignment.CENTER
            ),

            alignment=(
                ft.MainAxisAlignment.CENTER
            ),

            spacing=15,
        ),

        # FIXED SIZE
        width=INFO_BOX_WIDTH,
        height=INFO_BOX_HEIGHT,

        bgcolor=ft.Colors.GREY_900,

        border_radius=15,

        padding=20,
    )

    # =========================================================
    # ACCURACY
    # =========================================================

    accuracy_title = ft.Text(
        "Accuracy",
        size=22,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.WHITE,
        text_align=ft.TextAlign.CENTER,
    )

    accuracy_value = ft.Text(
        "0.00%",
        size=30,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.WHITE,
        text_align=ft.TextAlign.CENTER,
    )

    accuracy_container = ft.Container(
        content=ft.Column(
            [
                accuracy_title,
                accuracy_value,
            ],

            horizontal_alignment=(
                ft.CrossAxisAlignment.CENTER
            ),

            alignment=(
                ft.MainAxisAlignment.CENTER
            ),

            spacing=15,
        ),

        # FIXED SIZE
        width=INFO_BOX_WIDTH,
        height=INFO_BOX_HEIGHT,

        bgcolor=ft.Colors.GREY_900,

        border_radius=15,

        padding=20,
    )

    # =========================================================
    # INFORMATION PANEL
    # =========================================================

    information_panel = ft.Container(
        content=ft.Column(
            [
                status_container,
                accuracy_container,
            ],

            horizontal_alignment=(
                ft.CrossAxisAlignment.CENTER
            ),

            spacing=20,
        ),

        width=INFO_BOX_WIDTH,

        padding=10,

        alignment=ft.Alignment.CENTER,
    )

    # =========================================================
    # CAMERA CALLBACK
    # =========================================================

    def on_result(
        jpeg_data: bytes,
        top1_name,
        top1_conf,
    ):

        # -----------------------------------------------------
        # UPDATE CAMERA
        # -----------------------------------------------------

        video.src = jpeg_data
        video.update()

        # -----------------------------------------------------
        # UPDATE STATUS LABEL
        # -----------------------------------------------------

        label = str(top1_name)

        status_value.value = label

        # -----------------------------------------------------
        # CHANGE STATUS COLOR
        # -----------------------------------------------------

        status_color = STATUS_COLORS.get(
            label,
            ft.Colors.GREY_700,
        )

        status_container.bgcolor = status_color

        status_value.color = ft.Colors.WHITE

        status_container.update()

        status_value.update()

        # -----------------------------------------------------
        # UPDATE ACCURACY
        # -----------------------------------------------------

        confidence = float(top1_conf)

        # If confidence is returned as 0-1
        if confidence <= 1:
            confidence *= 100

        accuracy_value.value = (
            f"{confidence:.2f}%"
        )

        accuracy_value.update()

    # =========================================================
    # ERROR CALLBACK
    # =========================================================

    async def on_error(message: str):

        print(f"Camera error: {message}")

        status_value.value = "Camera Error"

        status_container.bgcolor = ft.Colors.RED_900

        status_value.update()

        status_container.update()

    # =========================================================
    # CAMERA WORKER
    # =========================================================

    worker = CameraWorker(
        on_result=on_result,
        on_error=on_error,
    )

    # =========================================================
    # WINDOW CLOSE
    # =========================================================

    async def on_window_event(e: ft.WindowEvent):

        if e.type == ft.WindowEventType.CLOSE:

            # Release camera
            await worker.stop()

            await page.window.destroy()

    page.window.prevent_close = True

    page.window.on_event = on_window_event

    # =========================================================
    # MAIN LAYOUT
    # =========================================================

    page.add(
        ft.Column(
            [
                # -------------------------------------------------
                # TITLE
                # -------------------------------------------------

                ft.Container(
                    content=title,

                    padding=ft.Padding(
                        top=15,
                        bottom=15,
                        left=10,
                        right=10,
                    ),

                    alignment=ft.Alignment.CENTER,
                ),

                # -------------------------------------------------
                # CAMERA + INFORMATION
                # -------------------------------------------------

                ft.Row(
                    [
                        # Camera
                        video_container,

                        # Information
                        information_panel,
                    ],

                    expand=True,

                    spacing=10,
                ),
            ],

            expand=True,

            spacing=0,
        )
    )

    # =========================================================
    # START CAMERA AUTOMATICALLY
    # =========================================================

    worker.start(CAMERA_INDEX)


# =============================================================
# RUN
# =============================================================

if __name__ == "__main__":
    ft.run(main)
