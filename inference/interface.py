import os
import subprocess
import threading
import queue
import signal
import gradio as gr
import time
import tempfile
import glob
import sys
import re
import shutil  # Added to copy files

# -------------------------------------------------
# If you are using Conda, set these paths accordingly
CONDA_ACTIVATE_PATH = "/opt/conda/etc/profile.d/conda.sh"
CONDA_ENV_NAME = "pyenv"

PROJECT_DIR = "/workspace/YuE-Interface"
# Default Hugging Face models
DEFAULT_STAGE1_MODEL = "/workspace/models/YuE-s1-7B-anneal-en-cot"
DEFAULT_STAGE2_MODEL = "/workspace/models/YuE-s2-1B-general"
TOKENIZER_MODEL = "/workspace/YuE-Interface/inference/mm_tokenizer_v0.2_hf/tokenizer.model"

sys.path.append(os.path.join(f"{PROJECT_DIR}/inference", 'xcodec_mini_infer'))
sys.path.append(os.path.join(f"{PROJECT_DIR}/inference", 'xcodec_mini_infer', 'descriptaudiocodec'))

# Output directory
DEFAULT_OUTPUT_DIR = "/workspace/outputs"
os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

DEFAULT_INPUT_DIR = "/workspace/inputs"
os.makedirs(DEFAULT_INPUT_DIR, exist_ok=True)

lyrics_example_file = open(f"{PROJECT_DIR}/inference/prompt_examples/lyrics.txt",mode='r')
lyrics_example = lyrics_example_file.read()
lyrics_example_file.close()

genre_example_file = open(f"{PROJECT_DIR}/inference/prompt_examples/genre.txt",mode='r')
genre_example = genre_example_file.read()
genre_example_file.close()

# -------------------------------------------------

# Queues for logs and audio paths
log_queue = queue.Queue()
audio_path_queue = queue.Queue()

process_dict = {}
process_lock = threading.Lock()


custom_log_box_css = """
#log_box textarea {
    overflow-y: scroll;
    max-height: 400px;  /* Set a max height for the log box */
    white-space: pre-wrap;  /* Preserve line breaks and white spaces */
    border: 1px solid #ccc;
    padding: 10px;
    font-family: monospace;
    scrollbar-width: thin!important;
}

#file_explorer {
max-height: 374px!important;
}

#file_explorer .file-wrap {
max-height: 320px!important;
}
"""

def get_selected_file(file_paths):
    """
    Handles file selection and prepares it for download.
    """
    # Handle the case when file_paths is a string (single file)
    if isinstance(file_paths, str):
        if os.path.isdir(file_paths):
            return None, "Please select a single file and not a folder."
        if not os.path.exists(file_paths):
            return None, f"File not found: {file_paths}"
        if not file_paths.lower().endswith('.mp3'):
            return None, f"File is not in .mp3 format: {file_paths}"
        return file_paths, f"File '{os.path.basename(file_paths)}' ready for download."

    # Handle the case when file_paths is a list (multiple files)
    if isinstance(file_paths, list) and file_paths:
        file_path = file_paths[0]  # Use the first file
        if os.path.isdir(file_path):
            return None, "Please select a single file and not a folder."
        if not os.path.exists(file_path):
            return None, f"File not found: {file_path}"
        if not file_path.lower().endswith('.mp3'):
            return None, f"File is not in .mp3 format: {file_paths}"
        return file_path, f"File '{os.path.basename(file_paths)}' ready for download."

    return None, "Invalid or no file selected."

def read_subprocess_output(proc, log_queue, audio_path_queue):
    """Reads subprocess stdout line by line, placing them into log_queue and audio_path_queue."""
    for line in iter(proc.stdout.readline, b''):
        decoded_line = line.decode("utf-8", errors="replace")
        print(f"Subprocess output: {decoded_line}")  # Debugging
        log_queue.put(decoded_line)
        
        # Detect the line containing "Created mix:"
        if "Created mix:" in decoded_line:
            # Extract the audio path using regex
            match = re.search(r"Created mix:\s*(\S+)", decoded_line)
            if match:
                audio_path = match.group(1)
                print(f"Audio path found: {audio_path}")  # Debugging
                audio_path_queue.put(audio_path)
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
        return "Inference stopped successfully."
    except Exception as e:
        return f"Error stopping process: {str(e)}"


def generate_song(
    stage1_model,
    stage2_model,
    tokenizer_model,
    genre_txt_path,
    lyrics_txt_path,
    run_n_segments,
    stage2_batch_size,
    output_dir,
    cuda_idx,
    max_new_tokens,
    use_audio_prompt,
    audio_prompt_file,
    prompt_start_time,
    prompt_end_time
):
    """Spawns infer.py to generate music, capturing logs in real time."""
    os.makedirs(output_dir, exist_ok=True)

    # If using an audio prompt, copy the file to DEFAULT_INPUT_DIR
    if use_audio_prompt and audio_prompt_file is not None:
        # Check if audio_prompt_file is a valid path
        if isinstance(audio_prompt_file, str):
            audio_filename = os.path.basename(audio_prompt_file)
            # Replace all especial characters with '_' to avoid issues with the command
            audio_filename = re.sub(r"[^a-zA-Z0-9.]", "_", audio_filename)
            saved_audio_path = os.path.join(DEFAULT_INPUT_DIR, audio_filename)
            shutil.copy(audio_prompt_file, saved_audio_path)
        else:
            return "Invalid audio prompt file format.", None
    else:
        saved_audio_path = ""

    # Build base command with '-u' for unbuffered output
    cmd = [
        "python", "-u", f"{PROJECT_DIR}/inference/infer.py",  # Adicionado '-u' aqui
        "--stage1_model", f"'{stage1_model}'",
        "--stage2_model", f"'{stage2_model}'",
        "--tokenizer", f"'{tokenizer_model}'",
        "--genre_txt", f"'{genre_txt_path}'",
        "--lyrics_txt", f"'{lyrics_txt_path}'",
        "--run_n_segments", str(run_n_segments),
        "--stage2_batch_size", str(stage2_batch_size),
        "--output_dir", f"'{output_dir}'",
        "--cuda_idx", str(cuda_idx),
        "--max_new_tokens", str(max_new_tokens),
        "--basic_model_config", f"'{PROJECT_DIR}/inference/xcodec_mini_infer/final_ckpt/config.yaml'",
        "--resume_path", f"'{PROJECT_DIR}/inference/xcodec_mini_infer/final_ckpt/ckpt_00360000.pth'",
        "--config_path", f"'{PROJECT_DIR}/inference/xcodec_mini_infer/decoders/config.yaml'",
        "--vocal_decoder_path", f"'{PROJECT_DIR}/inference/xcodec_mini_infer/decoders/decoder_131000.pth'",
        "--inst_decoder_path", f"'{PROJECT_DIR}/inference/xcodec_mini_infer/decoders/decoder_151000.pth'"
    ]
    
    if use_audio_prompt and saved_audio_path:
        cmd += [
            "--use_audio_prompt",
            "--audio_prompt_path", f"'{saved_audio_path}'",
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
    print(final_cmd_str)
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
    thread = threading.Thread(target=read_subprocess_output, args=(proc, log_queue, audio_path_queue), daemon=True)
    thread.start()

    return f"Inference started. Outputs will be saved in {output_dir}...", proc.pid


def update_logs(current_logs):
    """Pull all new lines from log_queue and append to current_logs."""
    new_text = ""
    while not log_queue.empty():
        new_text += log_queue.get()
    return current_logs + new_text


def build_gradio_interface():
    theme = gr.themes.Base()
    with gr.Blocks(title="YuE: Open Full-song Generation Foundation Model", theme=theme) as demo:
        gr.Markdown("# YuE - Gradio Interface\nEnter your Genre and Lyrics, then generate & listen!")

        with gr.Column():
            stage1_model = gr.Textbox(
                label="Stage1 Model",
                value=DEFAULT_STAGE1_MODEL,
                info="The model checkpoint path or identifier for the Stage 1 model."
            )
            stage2_model = gr.Textbox(
                label="Stage2 Model",
                value=DEFAULT_STAGE2_MODEL,
                info="The model checkpoint path or identifier for the Stage 2 model."
            )
            tokenizer_model = gr.Textbox(
                label="Tokenizer Model",
                value=TOKENIZER_MODEL,
                info="he model tokenizer path"
            )

            # Textboxes for genre and lyrics
            genre_textarea = gr.Textbox(
                label="Genre Text",
                lines=4,
                placeholder="Example: [Genre] inspiring female uplifting pop airy vocal...",
                info="Text containing genre tags that describe the musical style or characteristics (e.g., instrumental, genre, mood, vocal timbre, vocal gender). This is used as part of the generation prompt.",
                value=genre_example
            )
            lyrics_textarea = gr.Textbox(
                label="Lyrics Text",
                lines=4,
                placeholder="Type the lyrics here...",
                info="Text containing the lyrics for the music generation. These lyrics will be processed and split into structured segments to guide the generation process.",
                value=lyrics_example
            )

            run_n_segments = gr.Number(
                label="Number of Segments",
                value=2,
                precision=0,
                info="The number of segments to process during the generation."
            )
            stage2_batch_size = gr.Number(
                label="Stage2 Batch Size",
                value=4,
                precision=0,
                info="The batch size used in Stage 2 inference."
            )
            output_dir = gr.Textbox(
                label="Output Directory",
                value=DEFAULT_OUTPUT_DIR,
                info="The directory where generated outputs will be saved."
            )
            cuda_idx = gr.Number(
                label="CUDA Index",
                value=0,
                precision=0
            )
            max_new_tokens = gr.Number(
                label="Max New Tokens",
                value=3000,
                precision=0,
                info="The maximum number of new tokens to generate in one pass during text generation."
            )

            use_audio_prompt = gr.Checkbox(
                label="Use Audio Prompt?",
                value=False,
                info="If set, the model will use an audio file as a prompt during generation."
            )
            audio_prompt_file = gr.File(
                label="Upload Audio Prompt",
                file_types=["audio"],
                visible=False,
                file_count="single",  # Ensure that only one file is uploaded,
            )
            prompt_start_time = gr.Number(
                label="Prompt Start Time (s)",
                value=0,
                visible=False,
                info="The start time in seconds to extract the audio prompt from the given audio file."
            )
            prompt_end_time = gr.Number(
                label="Prompt End Time (s)",
                value=30,
                visible=False,
                info="The end time in seconds to extract the audio prompt from the given audio file."
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
                outputs=[audio_prompt_file, prompt_start_time, prompt_end_time]
            )

            generate_button = gr.Button("Generate Music")
            stop_button = gr.Button("Stop", visible=False)

            log_box = gr.Textbox(
                label="Logs",
                value="",
                lines=20,
                max_lines=30,
                interactive=False
            )
            
            explorer = gr.FileExplorer(root_dir=DEFAULT_OUTPUT_DIR, interactive=True, label="File Explorer", file_count="single", elem_id="file_explorer")     
            with gr.Column():   
                gr.Markdown("### Select a single file from the file explorer for download.")
            
            audio_status = gr.Textbox(label="File Status", interactive=False)
             
            # Section to show audio and allow download
            audio_player = gr.Audio(
                label="Generated Audio",
                type="filepath",
                value=None,
                interactive=False
            )
            
            # Event: When a file is selected in the explorer
            explorer.change(
                fn=get_selected_file,
                inputs=[explorer],
                outputs=[audio_player, audio_status],
            )

        # Hidden states
        generation_pid = gr.State(None)
        current_audio_path = gr.State(None)

        def on_generate_click(
            stage1_model,
            stage2_model,
            tokenizer_model,
            genre_text,
            lyrics_text,
            run_n_segments,
            stage2_batch_size,
            output_dir,
            cuda_idx,
            max_new_tokens,
            use_audio_prompt,
            audio_prompt_file,
            prompt_start_time,
            prompt_end_time
        ):
            """Triggered when user clicks 'Generate Music'."""
            # Check if a process is already running
            with process_lock:
                if process_dict:
                    return (
                        "Another process is running. Please stop it before starting a new one.",
                        None,
                        gr.update(visible=True),
                        gr.update(visible=False),
                        None,
                        None
                    )

            # Writes genre_text and lyrics_text to temporary .txt files
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
                tokenizer_model,
                genre_tmp_path,
                lyrics_tmp_path,
                run_n_segments,
                stage2_batch_size,
                output_dir,
                cuda_idx,
                max_new_tokens,
                use_audio_prompt,
                audio_prompt_file,
                prompt_start_time,
                prompt_end_time
            )
            # If the generation started successfully, hide "Generate" and show "Stop"
            if pid:
                return (msg, pid, gr.update(visible=False), gr.update(visible=True))
            else:
                return (msg, None, gr.update(visible=True), gr.update(visible=False))

        generate_button.click(
            fn=on_generate_click,
            inputs=[
                stage1_model,
                stage2_model,
                tokenizer_model,
                genre_textarea,
                lyrics_textarea,
                run_n_segments,
                stage2_batch_size,
                output_dir,
                cuda_idx,
                max_new_tokens,
                use_audio_prompt,
                audio_prompt_file,
                prompt_start_time,
                prompt_end_time
            ],
            outputs=[log_box, generation_pid, generate_button, stop_button]
        )

        def on_stop_click(pid):
            """Triggered when the user clicks 'Stop'."""
            status = stop_generation(pid)
            return (status, None, gr.update(visible=True), gr.update(visible=False))

        stop_button.click(
            fn=on_stop_click,
            inputs=[generation_pid],
            outputs=[log_box, generation_pid, generate_button, stop_button]
        )

        last_log_update = gr.State("")
        last_audio_update = gr.State(None)

        def refresh_state(log_text, pid, old_audio, last_log, last_audio):
            # Collect all new logs
            new_logs = ""
            while not log_queue.empty():
                new_logs += log_queue.get()
            
            # Collect new audio if available
            new_audio = old_audio
            while not audio_path_queue.empty():
                new_audio = audio_path_queue.get()

            # Check for real changes
            updated_log = log_text + new_logs if new_logs else log_text
            has_log_changes = updated_log != last_log
            has_audio_changes = new_audio != last_audio and new_audio is not None

            # Return gr.update() for fields that have changed, else no update
            return (
                updated_log if has_log_changes else gr.update(),  # log_box
                new_audio if has_audio_changes else gr.update(),  # current_audio_path
                updated_log if has_log_changes else last_log,     # last_log_update
                new_audio if has_audio_changes else last_audio    # last_audio_update
            )

        log_timer = gr.Timer(0.5, active=False)

        log_timer.tick(
            fn=refresh_state,
            inputs=[log_box, generation_pid, current_audio_path, last_log_update, last_audio_update],
            outputs=[log_box, current_audio_path, last_log_update, last_audio_update]
        )

        def activate_timer():
            return gr.update(active=True)

        generate_button.click(fn=activate_timer, outputs=[log_timer])
        stop_button.click(fn=activate_timer, outputs=[log_timer])
        
        def deactivate_timer():
            return gr.update(active=False)
        
        stop_button.click(fn=deactivate_timer, outputs=[log_timer])
        return demo


if __name__ == "__main__":
    interface = build_gradio_interface()
    # Adjust the port as needed
    interface.launch(server_name="0.0.0.0", server_port=7860, allowed_paths=["/workspace", ".", os.getcwd()])
