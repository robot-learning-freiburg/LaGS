# Copyright (c) 2026 Robert Bosch GmbH. All rights reserved.
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Viser server plumbing shared by the occupancy figure scripts.

Lighting + environment GUI, a frame navigator (slider / prev / next), and
screenshot helpers. The navigator is renderer-agnostic: it calls a
``render_fn(frame_idx)`` supplied by each entrypoint, so the same navigation and
screenshot UI drives the prediction and annotation figures alike.
"""

import logging
import re
from pathlib import Path
from typing import Callable

import imageio.v3 as iio
import numpy as np
import viser
from scipy.spatial.transform import Rotation as R

log = logging.getLogger(__name__)


def setup_lighting(server: viser.ViserServer) -> None:
    """Lighting + environment-map GUI controls."""
    server.scene.configure_environment_map(
        hdri="apartment",
        background=False,
        background_blurriness=0.5,
        environment_intensity=0.4,
        background_intensity=0.3,
    )

    with server.gui.add_folder("Default Lights"):
        gui_lights = server.gui.add_checkbox("Enable", initial_value=True)
        gui_shadows = server.gui.add_checkbox("Shadows", initial_value=True)

        def _update_lights(_):
            server.scene.configure_default_lights(gui_lights.value, gui_shadows.value)

        gui_lights.on_update(_update_lights)
        gui_shadows.on_update(_update_lights)

    with server.gui.add_folder("Environment"):
        gui_preset = server.gui.add_dropdown(
            "Preset",
            (
                "none",
                "apartment",
                "city",
                "dawn",
                "forest",
                "lobby",
                "night",
                "park",
                "studio",
                "sunset",
                "warehouse",
            ),
            initial_value="apartment",
        )
        gui_bg_show = server.gui.add_checkbox("Show Background", initial_value=False)
        gui_bg_intensity = server.gui.add_slider(
            "Background Intensity", min=0.0, max=1.0, step=0.01, initial_value=0.3
        )
        gui_env_intensity = server.gui.add_slider(
            "Environment Intensity", min=0.0, max=1.0, step=0.01, initial_value=0.4
        )

        def _update_env(_):
            server.scene.configure_environment_map(
                hdri=gui_preset.value if gui_preset.value != "none" else None,
                background=gui_bg_show.value,
                background_blurriness=0.5,
                background_intensity=gui_bg_intensity.value,
                environment_intensity=gui_env_intensity.value,
            )

        for handle in (gui_preset, gui_bg_show, gui_bg_intensity, gui_env_intensity):
            handle.on_update(_update_env)
        _update_env(None)


class FrameNavigator:
    """Navigate a sequence's frames, delegating drawing to ``render_fn``."""

    def __init__(
        self,
        server: viser.ViserServer,
        num_frames: int,
        render_fn: Callable[[int], None],
        scene_name: str,
        output_dir: Path,
    ):
        # pylint: disable=too-many-arguments
        self.server = server
        self.num_frames = num_frames
        self.render_fn = render_fn
        self.scene_name = scene_name
        self.output_dir = output_dir
        self.current_frame_idx = 0

    def goto_frame(self, frame_idx: int) -> None:
        if 0 <= frame_idx < self.num_frames:
            self.current_frame_idx = frame_idx
            self.render_fn(frame_idx)

    def next_frame(self) -> None:
        self.goto_frame((self.current_frame_idx + 1) % self.num_frames)

    def prev_frame(self) -> None:
        self.goto_frame((self.current_frame_idx - 1) % self.num_frames)


def setup_navigation_controls(
    server: viser.ViserServer, navigator: FrameNavigator
) -> None:
    """Frame slider, prev/next, and screenshot buttons wired to ``navigator``."""
    with server.gui.add_folder("Navigation"):
        gui_slider = server.gui.add_slider(
            "Frame",
            min=0,
            max=max(navigator.num_frames - 1, 0),
            step=1,
            initial_value=0,
        )
        gui_info = server.gui.add_text(
            "Frame Info", initial_value=f"1 / {navigator.num_frames}", disabled=True
        )
        gui_prev = server.gui.add_button("Previous")
        gui_next = server.gui.add_button("Next")
        gui_shot = server.gui.add_button("Screenshot (Top-down, all frames)")
        gui_shot_view = server.gui.add_button("Screenshot (Current view, all frames)")

        def _sync():
            gui_slider.value = navigator.current_frame_idx
            gui_info.value = (
                f"{navigator.current_frame_idx + 1} / {navigator.num_frames}"
            )

        @gui_slider.on_update
        def _on_slider(_):
            navigator.goto_frame(gui_slider.value)
            gui_info.value = (
                f"{navigator.current_frame_idx + 1} / {navigator.num_frames}"
            )

        @gui_prev.on_click
        def _on_prev(_):
            navigator.prev_frame()
            _sync()

        @gui_next.on_click
        def _on_next(_):
            navigator.next_frame()
            _sync()

        _wire_screenshot_button(gui_shot, navigator, take_topdown_screenshot, _sync)
        _wire_screenshot_button(
            gui_shot_view, navigator, take_current_view_screenshot, _sync
        )


def _wire_screenshot_button(button, navigator, capture, sync):
    """Capture ``capture(...)`` for every frame, restoring the current one after."""

    @button.on_click
    def _on_click(event: viser.GuiEvent):
        client = event.client
        if client is None:
            log.warning("No client connected")
            return
        original = navigator.current_frame_idx
        log.info(
            "Capturing %d screenshots for scene %s...",
            navigator.num_frames,
            navigator.scene_name,
        )
        for frame_idx in range(navigator.num_frames):
            navigator.goto_frame(frame_idx)
            capture(client, navigator.scene_name, frame_idx, navigator.output_dir)
            sync()
        navigator.goto_frame(original)
        sync()
        log.info("Completed capturing %d screenshots", navigator.num_frames)


def _screenshot_path(output_dir: Path, scene_name: str, frame_idx: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(scene_name))
    return output_dir / f"scene_{safe}_frame_{frame_idx:04d}.png"


def take_topdown_screenshot(
    client: viser.ClientHandle,
    scene_name: str,
    frame_idx: int,
    output_dir: Path = Path("screenshots"),
) -> None:
    """Render a fixed top-down view (driving direction up) and save it."""
    rotation = R.from_euler("xz", [180, -90], degrees=True).as_quat()  # [x, y, z, w]
    camera_wxyz = (rotation[3], rotation[0], rotation[1], rotation[2])
    image = client.get_render(
        height=1440,
        width=1440,
        position=(0.0, 0.0, 7.5),
        wxyz=camera_wxyz,
        fov=np.radians(60),
        transport_format="jpeg",
    )
    path = _screenshot_path(output_dir, scene_name, frame_idx)
    iio.imwrite(path, image)
    log.info("Screenshot saved to %s", path)


def take_current_view_screenshot(
    client: viser.ClientHandle,
    scene_name: str,
    frame_idx: int,
    output_dir: Path = Path("screenshots"),
) -> None:
    """Render from the client's current camera and save it."""
    image = client.get_render(height=1080, width=1920, transport_format="jpeg")
    path = _screenshot_path(output_dir, scene_name, frame_idx)
    iio.imwrite(path, image)
    log.info("Screenshot saved to %s", path)
