import os
import subprocess
import threading
import queue
import signal
import gradio as gr
import time
import tempfile
import glob

# -------------------------------------------------
# If you are using Conda, set these paths accordingly
CONDA_ACTIVATE_PATH = "/opt/conda/etc/profile.d/conda.sh"
CONDA_ENV_NAME = "pyenv"

# Default Hugging Face models
DEFAULT_STAGE1_MODEL = "/workspace/models/YuE-s1-7B-anneal-en-cot"
DEFAULT_STAGE2_MODEL = "/workspace/models/YuE-s2-1B-general"

# Output directory
DEFAULT_OUTPUT_DIR = "/workspace/outputs"

# -------------------------------------------------

process_dict = {}
process_lock = threading.Lock()
log_queue = queue.Queue()


def read_subprocess_output(proc, log_queue):
    """Reads subprocess stdout line by line, placing them into log_queue."""
    for line in iter(proc.stdout.readline, b''):
        decoded_line = line.decode("utf-8", errors="replace")
        log_queue.put(decoded_line)
    proc.stdout.close()
    proc.wait()
    with process_lock:
        if proc.pid in process_dict:
            del process_dict[proc.pid]


def stop_generation(pid):
    """Send signals to stop the subprocess if running."""
    if pid is None:
        return "No process is running."
    with process_lock:
        proc = process_dict.get(pid)
    if not proc:
        return "No process found or it has already stopped."

    if proc.poll() is not None:
        return "Process already finished."

    try:
        # Send SIGTERM first
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        time.sleep(2)
        if proc.poll() is None:
            # If still running, force kill
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        with process_lock:
            if pid in process_dict:
                del process_dict[pid]
        return "Inference stopped."
    except Exception as e:
        return f"Error stopping process: {str(e)}"


def generate_song(
    stage1_model,
    stage2_model,
    genre_txt_path,
    lyrics_txt_path,
    run_n_segments,
    stage2_batch_size,
    output_dir,
    cuda_idx,
    max_new_tokens,
    use_audio_prompt,
    audio_prompt_path,
    prompt_start_time,
    prompt_end_time
):
    """Spawns infer.py to generate music, capturing logs in real time."""
    os.makedirs(output_dir, exist_ok=True)

    # Build base command
    cmd = [
        "python", "infer.py",
        "--stage1_model", stage1_model,
        "--stage2_model", stage2_model,
        "--genre_txt", genre_txt_path,
        "--lyrics_txt", lyrics_txt_path,
        "--run_n_segments", str(run_n_segments),
        "--stage2_batch_size", str(stage2_batch_size),
        "--output_dir", output_dir,
        "--cuda_idx", str(cuda_idx),
        "--max_new_tokens", str(max_new_tokens),
    ]

    if use_audio_prompt and audio_prompt_path:
        cmd += [
            "--use_audio_prompt",
            "--audio_prompt_path", audio_prompt_path,
            "--prompt_start_time", str(prompt_start_time),
            "--prompt_end_time", str(prompt_end_time)
        ]

    # If using conda, wrap the command
    if os.path.isfile(CONDA_ACTIVATE_PATH):
        prefix_cmd = (
            f"bash -c 'source {CONDA_ACTIVATE_PATH} && "
            f"conda activate {CONDA_ENV_NAME} && "
        )
        suffix_cmd = "'"
        final_cmd_str = prefix_cmd + " ".join(cmd) + suffix_cmd
    else:
        final_cmd_str = " ".join(cmd)

    proc = subprocess.Popen(
        final_cmd_str,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid  # allows signal handling
    )

    with process_lock:
        process_dict[proc.pid] = proc

    # Thread to read logs
    thread = threading.Thread(target=read_subprocess_output, args=(proc, log_queue), daemon=True)
    thread.start()

    return f"Inference started. Outputs will be saved in {output_dir}...", proc.pid


def update_logs(current_logs):
    """Pull all new lines from log_queue and append to current_logs."""
    new_text = ""
    while not log_queue.empty():
        new_text += log_queue.get()
    return current_logs + new_text


def find_newest_wav(output_dir):
    """Return the newest .wav file in output_dir, or None if not found."""
    wav_files = glob.glob(os.path.join(output_dir, "*.wav"))
    if not wav_files:
        return None
    # Sort by creation time descending
    newest = max(wav_files, key=os.path.getctime)
    return newest


def build_gradio_interface():
    with gr.Blocks(title="YuE Song Generation Interface") as demo:
        gr.Markdown("# YuE Song Generation\nWrite your Genre and Lyrics in TextAreas, then generate & listen!")

        with gr.Row():
            with gr.Column():
                stage1_model = gr.Textbox(
                    label="Stage1 Model (HF repo)",
                    value=DEFAULT_STAGE1_MODEL
                )
                stage2_model = gr.Textbox(
                    label="Stage2 Model (HF repo)",
                    value=DEFAULT_STAGE2_MODEL
                )

                # TextAreas for genre and lyrics
                genre_textarea = gr.Textbox(
                    label="Genre text",
                    lines=4,
                    placeholder="Example: [Genre] inspiring female uplifting pop airy vocal..."
                )
                lyrics_textarea = gr.Textbox(
                    label="Lyrics text",
                    lines=4,
                    placeholder="Type the lyrics here..."
                )

                run_n_segments = gr.Number(
                    label="Number of segments",
                    value=2,
                    precision=0
                )
                stage2_batch_size = gr.Number(
                    label="Stage2 Batch Size",
                    value=4,
                    precision=0
                )
                output_dir = gr.Textbox(
                    label="Output Directory",
                    value=DEFAULT_OUTPUT_DIR
                )
                cuda_idx = gr.Number(
                    label="CUDA Index",
                    value=0,
                    precision=0
                )
                max_new_tokens = gr.Number(
                    label="Max New Tokens",
                    value=3000,
                    precision=0
                )

                use_audio_prompt = gr.Checkbox(
                    label="Use Audio Prompt?",
                    value=False
                )
                audio_prompt_path = gr.Textbox(
                    label="Audio Prompt Path",
                    value="",
                    visible=False
                )
                prompt_start_time = gr.Number(
                    label="Prompt Start Time (s)",
                    value=0,
                    visible=False
                )
                prompt_end_time = gr.Number(
                    label="Prompt End Time (s)",
                    value=30,
                    visible=False
                )

                def toggle_audio_prompt(checked):
                    return [
                        gr.update(visible=checked),
                        gr.update(visible=checked),
                        gr.update(visible=checked)
                    ]

                use_audio_prompt.change(
                    fn=toggle_audio_prompt,
                    inputs=use_audio_prompt,
                    outputs=[audio_prompt_path, prompt_start_time, prompt_end_time]
                )

                generate_button = gr.Button("Generate Music")
                stop_button = gr.Button("Stop", visible=False)

            with gr.Column():
                log_box = gr.Textbox(
                    label="Logs",
                    value="",
                    lines=20,
                    max_lines=30,
                    interactive=False
                )
                # Section to show audio and let user download
                audio_player = gr.Audio(
                    label="Generated Audio",
                    source="filepath",
                    type="filepath",
                    value=None,
                    interactive=False
                )
                audio_downloader = gr.File(
                    label="Download Generated Audio",
                    interactive=False,
                    value=None
                )

        # Hidden states
        generation_pid = gr.State(None)
        current_audio_path = gr.State(None)

        def on_generate_click(
            stage1_model,
            stage2_model,
            genre_text,
            lyrics_text,
            run_n_segments,
            stage2_batch_size,
            output_dir,
            cuda_idx,
            max_new_tokens,
            use_audio_prompt,
            audio_prompt_path,
            prompt_start_time,
            prompt_end_time
        ):
            """Triggered when user clicks 'Generate Music'."""
            # Check if a process is already running
            with process_lock:
                if process_dict:
                    return (
                        "Another process is running. Stop it before starting a new one.",
                        None,
                        gr.update(visible=True),
                        gr.update(visible=False),
                        None,
                        None
                    )

            # Write the genre_text and lyrics_text into temporary .txt files:
            import tempfile
            def write_temp_file(content, suffix=".txt"):
                fd, path = tempfile.mkstemp(suffix=suffix)
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(content)
                return path

            genre_tmp_path = write_temp_file(genre_text, ".txt")
            lyrics_tmp_path = write_temp_file(lyrics_text, ".txt")

            msg, pid = generate_song(
                stage1_model,
                stage2_model,
                genre_tmp_path,
                lyrics_tmp_path,
                run_n_segments,
                stage2_batch_size,
                output_dir,
                cuda_idx,
                max_new_tokens,
                use_audio_prompt,
                audio_prompt_path,
                prompt_start_time,
                prompt_end_time
            )
            # If generation started successfully, hide "Generate" and show "Stop"
            if pid:
                return (msg, pid, gr.update(visible=False), gr.update(visible=True), None, None)
            else:
                return (msg, None, gr.update(visible=True), gr.update(visible=False), None, None)

        generate_button.click(
            fn=on_generate_click,
            inputs=[
                stage1_model,
                stage2_model,
                genre_textarea,
                lyrics_textarea,
                run_n_segments,
                stage2_batch_size,
                output_dir,
                cuda_idx,
                max_new_tokens,
                use_audio_prompt,
                audio_prompt_path,
                prompt_start_time,
                prompt_end_time
            ],
            outputs=[log_box, generation_pid, generate_button, stop_button, audio_player, audio_downloader]
        )

        def on_stop_click(pid):
            """Triggered when user clicks 'Stop'."""
            status = stop_generation(pid)
            return (status, None, gr.update(visible=True), gr.update(visible=False), None, None)

        stop_button.click(
            fn=on_stop_click,
            inputs=[generation_pid],
            outputs=[log_box, generation_pid, generate_button, stop_button, audio_player, audio_downloader]
        )

        # Timer to update logs + check if process ended
        def refresh_state(log_text, pid, old_audio):
            # 1) Update logs
            updated_logs = update_logs(log_text)

            # 2) If the process is done (pid not in process_dict), load the newest wav
            newest_audio = old_audio
            with process_lock:
                # If process has ended (pid not in dict or pid is None)
                if pid and pid not in process_dict:
                    # The generation must be complete
                    found_wav = find_newest_wav(output_dir.value)
                    if found_wav and found_wav != old_audio:
                        newest_audio = found_wav

            return updated_logs, newest_audio

        # This timer triggers every second, updating logs and the audio path if generation is done
        log_timer = gr.Timer(interval=1.0)

        log_timer_fn = log_timer.tick(
            fn=refresh_state,
            inputs=[log_box, generation_pid, current_audio_path],
            outputs=[log_box, current_audio_path]
        )

        # Then we chain a function to update the UI Audio and Download if there's a new file
        def update_audio_player(audio_path):
            # If no path, do nothing
            if not audio_path:
                return gr.update(), gr.update()
            # Otherwise, set the audio player and file for download
            return audio_path, audio_path

        log_timer_fn.then(
            fn=update_audio_player,
            inputs=[current_audio_path],
            outputs=[audio_player, audio_downloader]
        )

        # We start the timer only after user clicks "Generate" or "Stop"
        def activate_timer():
            return gr.update(active=True)

        def deactivate_timer():
            return gr.update(active=False)

        generate_button.click(fn=activate_timer, outputs=[log_timer])
        stop_button.click(fn=activate_timer, outputs=[log_timer])  # keep logs going

        # If you prefer stopping logs when user stops generation, uncomment below
        # stop_button.click(fn=deactivate_timer, outputs=[log_timer])

    return demo


if __name__ == "__main__":
    interface = build_gradio_interface()
    # Adjust port as needed
    interface.launch(server_name="0.0.0.0", server_port=7860)
