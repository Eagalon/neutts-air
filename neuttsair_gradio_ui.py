import os 
import torch
import soundfile as sf
import numpy as np
import pyaudio
import gradio as gr
import nltk
from neuttsair.neutts import NeuTTSAir
from neucodec import NeuCodec

# Download NLTK sentence tokenizer
try:
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK punkt tokenizer: {str(e)}")

def encode_reference_audio(ref_audio_path, output_path="encoded_reference.pt"):
    """Pre-encode a reference audio file into codes."""
    if not ref_audio_path:
        return None, "Please upload a reference audio file."
    if not output_path.endswith(".pt"):
        return None, "Output path must end with .pt"
    
    try:
        codec = NeuCodec.from_pretrained("neuphonic/neucodec")
        codec.eval().to("cpu")
        wav, _ = sf.read(ref_audio_path, dtype='float32', always_2d=False)
        if wav.ndim > 1:
            wav = wav[:, 0]  # Convert to mono if stereo
        wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
        ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        torch.save(ref_codes, output_path)
        return output_path, f"Reference audio encoded and saved to {output_path}"
    except Exception as e:
        return None, f"Error encoding reference audio: {str(e)}"

def generate_audio(input_text, ref_audio, ref_text, ref_codes, backbone, use_onnx, stream_output, chunked_processing):
    """Generate or stream audio using NeuTTSAir, with option for chunked processing."""
    if not input_text or not input_text.strip():
        return None, "Please provide non-empty input text."
    if not ref_audio and not ref_codes:
        return None, "Please provide either a reference audio file or pre-encoded reference codes."
    
    try:
        # Initialize NeuTTSAir
        codec_repo = "neuphonic/neucodec-onnx-decoder" if use_onnx else "neuphonic/neucodec"
        tts = NeuTTSAir(
            backbone_repo=backbone,
            backbone_device="cpu",
            codec_repo=codec_repo,
            codec_device="cpu"
        )

        # Handle reference text
        if ref_text and os.path.exists(ref_text):
            with open(ref_text, "r") as f:
                ref_text = f.read().strip()
        elif not ref_text:
            ref_text = ""  # Default to empty string if no reference text provided

        # Handle reference codes
        if ref_codes and os.path.exists(ref_codes):
            ref_codes_data = torch.load(ref_codes)
        elif ref_audio:
            ref_codes_data = tts.encode_reference(ref_audio)
        else:
            ref_codes_data = None

        if ref_codes_data is None:
            return None, "Failed to load or encode reference audio."

        # Split text into sentences if chunked processing is enabled
        if chunked_processing and not stream_output:
            try:
                sentences = nltk.sent_tokenize(input_text)
            except Exception as e:
                return None, f"Error tokenizing text into sentences: {str(e)}"
            
            if not sentences:
                return None, "No valid sentences detected in input text."
            
            audio_chunks = []
            for i, sentence in enumerate(sentences):
                sentence = sentence.strip()
                if not sentence:
                    continue
                try:
                    wav = tts.infer(sentence, ref_codes_data, ref_text)
                    if wav is not None and len(wav) > 0:
                        audio_chunks.append(wav)
                    else:
                        return None, f"Failed to generate audio for sentence {i+1}: Empty or invalid audio output."
                except Exception as e:
                    return None, f"Error generating audio for sentence {i+1}: {str(e)}"
            
            if not audio_chunks:
                return None, "No valid audio chunks generated from input text."
            
            # Combine audio chunks
            try:
                combined_wav = np.concatenate(audio_chunks)
                output_path = "output_chunked.wav"
                sf.write(output_path, combined_wav, 24000)
                return output_path, f"Audio generated for {len(audio_chunks)} sentences and saved to {output_path}"
            except Exception as e:
                return None, f"Error combining audio chunks: {str(e)}"
        elif stream_output:
            if backbone not in ["neuphonic/neutts-air-q4-gguf", "neuphonic/neutts-air-q8-gguf"]:
                return None, "Streaming is only supported with GGUF backbones."
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=24000,
                output=True
            )
            try:
                if chunked_processing:
                    try:
                        sentences = nltk.sent_tokenize(input_text)
                    except Exception as e:
                        return None, f"Error tokenizing text into sentences: {str(e)}"
                    
                    if not sentences:
                        return None, "No valid sentences detected in input text."
                    
                    for i, sentence in enumerate(sentences):
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        try:
                            for chunk in tts.infer_stream(sentence, ref_codes_data, ref_text):
                                if chunk is not None and len(chunk) > 0:
                                    audio = (chunk * 32767).astype(np.int16)
                                    stream.write(audio.tobytes())
                                else:
                                    return None, f"Failed to stream audio for sentence {i+1}: Empty or invalid audio chunk."
                        except Exception as e:
                            return None, f"Error streaming audio for sentence {i+1}: {str(e)}"
                    return None, f"Streamed audio for {len(sentences)} sentences."
                else:
                    for chunk in tts.infer_stream(input_text, ref_codes_data, ref_text):
                        if chunk is not None and len(chunk) > 0:
                            audio = (chunk * 32767).astype(np.int16)
                            stream.write(audio.tobytes())
                        else:
                            return None, "Failed to stream audio: Empty or invalid audio chunk."
                    return None, "Audio streamed successfully."
            finally:
                stream.stop_stream()
                stream.close()
                p.terminate()
        else:
            # Non-chunked, non-streaming
            wav = tts.infer(input_text, ref_codes_data, ref_text)
            if wav is None or len(wav) == 0:
                return None, "Failed to generate audio: Empty or invalid audio output."
            output_path = "output.wav"
            sf.write(output_path, wav, 24000)
            return output_path, f"Audio saved to {output_path}"
    except Exception as e:
        return None, f"Error generating audio: {str(e)}"

def gradio_interface():
    """Create the Gradio web interface."""
    with gr.Blocks(title="NeuTTSAir Text-to-Speech Interface") as demo:
        gr.Markdown(
            """
            # NeuTTSAir Text-to-Speech Interface
            Convert text to speech using NeuTTSAir with options for reference audio encoding, low-latency ONNX decoding, streaming output, and chunked text processing.
            """
        )

        # Input Section
        with gr.Group(elem_id="input-section"):
            gr.Markdown("## Input Parameters", elem_id="input-heading")
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Enter text to convert to speech",
                lines=5,
                elem_id="input-text"
            )
            ref_audio = gr.Audio(
                label="Reference Audio",
                type="filepath",
                elem_id="ref-audio"
            )
            ref_text = gr.Textbox(
                label="Reference Text",
                placeholder="Enter reference text or path to a text file",
                elem_id="ref-text"
            )
            ref_codes = gr.File(
                label="Pre-encoded Reference Codes",
                file_types=[".pt"],
                elem_id="ref-codes"
            )
            backbone = gr.Dropdown(
                label="Backbone Model",
                choices=["neuphonic/neutts-air", "neuphonic/neutts-air-q4-gguf", "neuphonic/neutts-air-q8-gguf"],
                value="neuphonic/neutts-air-q4-gguf",
                elem_id="backbone"
            )
            use_onnx = gr.Checkbox(
                label="Use ONNX Decoder (Low Latency)",
                value=False,
                elem_id="use-onnx"
            )
            stream_output = gr.Checkbox(
                label="Stream Output (GGUF Backbones Only)",
                value=False,
                elem_id="stream-output"
            )
            chunked_processing = gr.Checkbox(
                label="Process Text in Sentence Chunks",
                value=False,
                elem_id="chunked-processing"
            )

        # Encode Reference Section
        with gr.Group(elem_id="encode-section"):
            gr.Markdown("## Encode Reference Audio", elem_id="encode-heading")
            encode_button = gr.Button("Encode Reference Audio", elem_id="encode-button")
            encode_output = gr.Textbox(
                label="Encoding Status",
                interactive=False,
                elem_id="encode-status"
            )
            encode_button.click(
                fn=encode_reference_audio,
                inputs=[ref_audio],
                outputs=[ref_codes, encode_output],
                api_name="encode_reference"
            )

        # Generate Audio Section
        with gr.Group(elem_id="generate-section"):
            gr.Markdown("## Generate Audio", elem_id="generate-heading")
            generate_button = gr.Button("Generate/Stream Audio", elem_id="generate-button")
            audio_output = gr.Audio(
                label="Generated Audio",
                type="filepath",
                interactive=False,
                elem_id="audio-output"
            )
            status_output = gr.Textbox(
                label="Generation Status",
                interactive=False,
                elem_id="status-output"
            )
            generate_button.click(
                fn=generate_audio,
                inputs=[input_text, ref_audio, ref_text, ref_codes, backbone, use_onnx, stream_output, chunked_processing],
                outputs=[audio_output, status_output],
                api_name="generate_audio"
            )

        # Accessibility Enhancements with CSS and JavaScript
        demo.css = """
        #input-section, #encode-section, #generate-section {
            margin-bottom: 20px;
            padding: 10px;
            border: 1px solid #ccc;
            border-radius: 5px;
        }
        #input-heading, #encode-heading, #generate-heading {
            font-size: 1.5em;
            margin-bottom: 10px;
        }
        button {
            font-size: 1em;
            padding: 10px;
        }
        .gr-textbox, .gr-audio, .gr-file, .gr-dropdown, .gr-checkbox {
            margin-bottom: 10px;
        }
        label {
            font-weight: bold;
            margin-bottom: 5px;
            display: block;
        }
        """

        # JavaScript to add ARIA attributes dynamically
        gr.HTML("""
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                // Add ARIA labels
                document.getElementById('input-text').setAttribute('aria-label', 'Text input for speech synthesis');
                document.getElementById('ref-audio').setAttribute('aria-label', 'Upload a reference audio file for voice cloning');
                document.getElementById('ref-text').setAttribute('aria-label', 'Reference text corresponding to the reference audio');
                document.getElementById('ref-codes').setAttribute('aria-label', 'Upload pre-encoded reference codes (.pt file)');
                document.getElementById('backbone').setAttribute('aria-label', 'Select the backbone model for TTS');
                document.getElementById('use-onnx').setAttribute('aria-label', 'Enable ONNX decoder for low-latency inference');
                document.getElementById('stream-output').setAttribute('aria-label', 'Enable streaming audio output');
                document.getElementById('chunked-processing').setAttribute('aria-label', 'Process text in sentence chunks for large inputs');
                document.getElementById('encode-button').setAttribute('aria-label', 'Encode reference audio into codes');
                document.getElementById('generate-button').setAttribute('aria-label', 'Generate or stream audio output');
                document.getElementById('encode-status').setAttribute('aria-label', 'Status of reference audio encoding');
                document.getElementById('audio-output').setAttribute('aria-label', 'Generated audio output');
                document.getElementById('status-output').setAttribute('aria-label', 'Status of audio generation or streaming');
                
                // Ensure focusable elements are accessible
                document.querySelectorAll('button, input, select, textarea').forEach(el => {
                    if (!el.hasAttribute('tabindex')) {
                        el.setAttribute('tabindex', '0');
                    }
                });
            });
        </script>
        """)

    return demo

if __name__ == "__main__":
    demo = gradio_interface()
    demo.launch()