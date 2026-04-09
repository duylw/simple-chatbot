import logging
import gradio as gr
import httpx
import urllib.parse
import unicodedata

logger = logging.getLogger(__name__)

API_BASE_URL = "http://localhost:8000"

async def get_response(query: str):
    """Fetch response from the Agentic RAG API and extract video links"""
    if not query.strip():
        return "Please enter a question.", []

    try:
        url = f"{API_BASE_URL}/agentic_ask/"
        params = {"question": query}

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(url, params=params)
            
            if response.status_code != 200:
                return f"**Error: API returned status {response.status_code}**\n\nDetails: {response.text}", []

            data = response.json()
            
            answer = data.get("answer", "No answer found.")
            sources = data.get("sources", [])
            
            # 1. Format the Markdown response
            formatted_response = answer
            
            # 2. Extract sources for the clickable table
            video_data = [] # Will hold [Video Name, Timestamp, URL]
            
            if sources:
                for doc in sources:
                    metadata = doc.get("metadata", {})
                    # Adjust 'video_name' based on what your vector DB actually stores
                    video_name = metadata.get("source", metadata.get("video_name", "Unknown Video"))
                    
                    # Ensure filename ends with .mp4 if it doesn't already
                    filename = video_name if video_name.endswith('.mp4') else f"{video_name}.mp4"
                    
                    # Normalize Unicode characters (crucial for Vietnamese accents)
                    filename = unicodedata.normalize('NFC', filename)
                    
                    # Add #t=timestamp for jumping to the specific time
                    timestamp = metadata.get("timestamp", 0)
                    video_url = f"{API_BASE_URL}/media/videos/{filename}#t={timestamp}"
                    
                    # Format timestamp as MM:SS for display
                    mins, secs = divmod(int(timestamp), 60)
                    time_display = f"{mins:02d}:{secs:02d}"
                    
                    video_data.append([video_name, time_display, video_url])

            return formatted_response, video_data

    except Exception as e:
        return f"Unexpected error: {str(e)}", []


def play_selected_video(evt: gr.SelectData, source_data):
    """Handler for when a user clicks a row in the sources DataFrame"""
    row_idx = evt.index[0]
    selected_url = source_data.iloc[row_idx, 2]
    
    # Put the 'src' directly in the <video> tag instead of a <source> child tag
    # This forces the browser player to reload when you click a different row
    html_player = f'''
    <video width="100%" controls autoplay src="{selected_url}">
      Your browser does not support the video tag.
    </video>
    '''
    return html_player


def create_gradio_interface():
    theme = gr.themes.Default(primary_hue="zinc", neutral_hue="slate")
    
    # Custom CSS to target the markdown response text
    custom_css = """
    .response-markdown {
        font-size: 1.3rem !important;
        line-height: 1.6 !important;
    }
    """

    with gr.Blocks(title="Agentic RAG Assistant", theme=theme, css=custom_css) as interface:
        gr.Markdown("# RAG Assistant")

        with gr.Row():
            with gr.Column(scale=1):
                # Request UI
                query_input = gr.Textbox(placeholder="Ask a question...")
                submit_btn = gr.Button("Search", variant="primary")
                
                # Response UI (apply the custom class here)
                response_output = gr.Markdown(
                    "_Awaiting your question..._", 
                    elem_classes=["response-markdown"],
                    latex_delimiters=[
                        {"left": "$$", "right": "$$", "display": True},   # Block math
                        {"left": "$", "right": "$", "display": False}     # Inline math (display: False means inline)
                    ]
                )
                
            with gr.Column(scale=1):
                # Video Player UI
                gr.Markdown("### Video Player")
                video_player = gr.HTML()

                # Interactive Sources Table
                gr.Markdown("### Sources (Click row to play video)")
                sources_df = gr.Dataframe(
                    headers=["Video Name", "Timestamp", "Hidden_URL"], 
                    datatype=["str", "str", "str"],
                    interactive=False,
                    wrap=True
                )

        # Triggers
        submit_btn.click(
            fn=get_response,
            inputs=[query_input],
            outputs=[response_output, sources_df],
        )

        query_input.submit(
            fn=get_response,
            inputs=[query_input],
            outputs=[response_output, sources_df],
        )

        # When a user clicks a row in the Dataframe, send the URL to the Video Player
        sources_df.select(
            fn=play_selected_video,
            inputs=[sources_df],
            outputs=[video_player]
        )

    return interface

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=False)