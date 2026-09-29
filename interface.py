

import asyncio
import base64
import math
import platform

import cv2
import flet as ft

import detection


# =============================================================
# CONFIGURATION
# =============================================================

CAMERA_INDEX = 1

# Reduced camera resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Fixed size for information boxes
INFO_BOX_WIDTH = 500
INFO_BOX_HEIGHT = 300

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
    "Breaking": "#2b3a06",
    "Ripe_first_stage": "#6b522d",
    "Ripe_second_stage": "#67502B",
    "Underripe": "#908c48",
    "Overripe": "#25031f",
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


        ok, raw_frame = cap.read()

        detector = detection.Avocado_detector()

        if not ok:
            return None
        # height, width = raw_frame.shape[:2]
        
        # print(f"Incoming frame resolution: {width} x {height}")

        cropped_frame = raw_frame[140:940,560:1360]
        # ---------------------------------------------------------
        # CLASSIFICATION
        # ---------------------------------------------------------

        frame, top1_name, top1_conf = detector.classify(
            cropped_frame
        )

        # ---------------------------------------------------------
        # ENCODE FRAME FOR FLET
        # ---------------------------------------------------------
        cv2.rectangle(
            raw_frame,
            (560,140),
            (1360,940),
            (255,255,255),
            thickness=10,
        )

        ok, buffer = cv2.imencode(
            ".jpg",
            raw_frame,
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

            # cap.set(
            #     cv2.CAP_PROP_FRAME_WIDTH,
            #     CAMERA_WIDTH,
            # )

            # cap.set(
            #     cv2.CAP_PROP_FRAME_HEIGHT,
            #     CAMERA_HEIGHT,
            # )

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
    


    # Remove all margins
    page.padding = 0

    # ---------------------------------------------------------
    # TRUE FULLSCREEN
    # ---------------------------------------------------------

    page.window.full_screen = True

    # =========================================================
    # TITLE
    # ===========================
    # =========================
    title = ft.Text(
        "Ripeness Classification",
        size=100,
        weight=ft.FontWeight.W_900,
        color="#356515",
        text_align=ft.TextAlign.CENTER,
    )

    title_2 = ft.Text(
        "Hass Avocado",
        size=50,
        weight=ft.FontWeight.W_900,
        color="#356515",
        text_align=ft.TextAlign.CENTER,
    )

    title_3 = ft.Column(
        [
            title,
            title_2,
        ],
        alignment=ft.MainAxisAlignment.CENTER,
        horizontal_alignment=ft.CrossAxisAlignment.CENTER,
        spacing=0,
    )
        

    # =========================================================
    # VIDEO
    # =========================================================

    video = ft.Image(
        src=PLACEHOLDER,
        fit=ft.BoxFit.CONTAIN,
        gapless_playback=True,
        expand=True,
        border_radius=20,
    )

    video_with_shadow = ft.Container(
        content=video,
        expand=True,
        border_radius=20,
        shadow=ft.BoxShadow(
            spread_radius=3,
            blur_radius=20,
            color= "#67502B",
            offset=ft.Offset(0, 5), # Desplazamiento x, y
        ),
    )

    video_container = ft.Container(
        content=video_with_shadow,
        expand=True,
        alignment=ft.Alignment.CENTER,
    )
    avocado_png = ft.Image(
        src="avocado.png",
        width=300,
        height=300,
        fit=ft.BoxFit.CONTAIN,
    )

    avocado_container = ft.Container(
        content=avocado_png,
        width=300,
        height=300,
        alignment=ft.Alignment.CENTER,
    )
    
    video_container = ft.Container(
        content=video,

        expand=True,

        # bgcolor=ft.Colors.BLACK,

        alignment=ft.Alignment.CENTER,

        border_radius=20,
    )

    # =========================================================
    # AVOCADO STATUS
    # =========================================================

    status_title = ft.Text(
        "Avocado status",
        size=40,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.WHITE,
        text_align=ft.TextAlign.CENTER,
    )

    status_value = ft.Text(
        "Waiting...",
        size=30,
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

        bgcolor=ft.Colors.GREY_700,

        border_radius=15,

        padding=20,

        shadow=ft.BoxShadow(
            spread_radius=3,
            blur_radius=20,
            color= "#67502B",
            offset=ft.Offset(0, 5), # Desplazamiento x, y
        ),
        margin=ft.Margin.only(top=30),
    )

    # =========================================================
    # ACCURACY
    # =========================================================

    accuracy_title = ft.Text(
        "Accuracy",
        size=40,
        weight=ft.FontWeight.BOLD,
        color=ft.Colors.WHITE,
        text_align=ft.TextAlign.CENTER,
    )

    accuracy_value = ft.Text(
        "*",
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

        bgcolor=ft.Colors.GREY_700,

        border_radius=15,

        padding=20,
        margin=ft.Margin.only(top=80),
        shadow=ft.BoxShadow(
            spread_radius=1,
            blur_radius=15,
            color= "#67502B",
            offset=ft.Offset(0, 5), # Desplazamiento x, y
        ),
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
        margin=ft.Margin.only(right=30)
    )
    
    # =========================================================
    # CAMERA CALLBACK
    # =========================================================

    def on_result(
        jpeg_data: bytes,
        top1_name,
        top1_conf,
    ):
        confidence = float(top1_conf)

        # If confidence is returned as 0-1
        if confidence <= 1:
            confidence *= 100
        
        accuracy_value.value = (
            f"{confidence:.2f}%" #if confidence > 80 else "*"
        )
        if accuracy_value.value == "*":
            accuracy_container.bgcolor = ft.Colors.GREY_700
        else:
            accuracy_container.bgcolor = "#D89465"

        accuracy_value.update()
        accuracy_container.update()
        # -----------------------------------------------------
        # UPDATE CAMERA
        # -----------------------------------------------------

        video.src = jpeg_data
        video.update()

        # -----------------------------------------------------
        # UPDATE STATUS LABEL
        # -----------------------------------------------------

        label = str(top1_name)

        status_value.value = label #if confidence > 80 else "Waiting..."

        # -----------------------------------------------------
        # CHANGE STATUS COLOR
        # -----------------------------------------------------

        status_color = STATUS_COLORS.get(
            label,
            ft.Colors.GREY_700,
        ) if confidence > 80 else ft.Colors.GREY_700

        status_container.bgcolor = status_color

        status_value.color = ft.Colors.WHITE

        status_container.update()

        status_value.update()

        # -----------------------------------------------------
        # UPDATE ACCURACY
        # -----------------------------------------------------



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
        ft.Container(
            expand=True,

            gradient=ft.LinearGradient(
                begin=ft.Alignment.TOP_LEFT,
                end=ft.Alignment(0.8, 1),
                tile_mode=ft.GradientTileMode.MIRROR,
                rotation=math.pi / 3,
                colors=[
                    "#D8D881",
                    "#C0B54B"
                ],
            ),

            content=ft.Column(
                [   
                    ft.Row(
                        [
                            ft.Container(
                                content=title_3,

                                padding=ft.Padding(
                                    top=15,
                                    bottom=0,
                                    left=10,
                                    right=10,
                                ),

                                alignment=ft.Alignment.CENTER,
                            ),

                            avocado_container,
                        ],

                        alignment=ft.MainAxisAlignment.CENTER,
                        vertical_alignment=ft.CrossAxisAlignment.CENTER,
                        spacing=20,
                    ),
                                    

                    ft.Row(
                        [
                            video_container,
                            information_panel,
                        ],

                        expand=True,
                        spacing=10,
                    ),
                ],

                expand=True,
                spacing=0,
            ),
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
