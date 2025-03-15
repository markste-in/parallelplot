import nbformat
from nbconvert import MarkdownExporter
import os

def convert_notebook_to_markdown(notebook_path, output_path=None):
    """
    Convert a Jupyter notebook to Markdown format
    
    Args:
        notebook_path (str): Path to the .ipynb file
        output_path (str, optional): Path to save the markdown file. 
                                     If None, uses the same name as the notebook with .md extension.
    
    Returns:
        str: Path to the created markdown file
    """
    # Default output path if none provided
    if output_path is None:
        base_name = os.path.splitext(notebook_path)[0]
        output_path = base_name + '.md'
    
    # Load the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
    
    # Initialize the markdown exporter
    md_exporter = MarkdownExporter()
    
    # Convert the notebook to markdown
    (body, resources) = md_exporter.from_notebook_node(nb)
    
    # Write the markdown content to a file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(body)
    
    # If there are resources (like images), save them in a subfolder
    if resources.get('outputs'):
        # Create a subfolder for images
        base_name = os.path.splitext(os.path.basename(output_path))[0]
        images_dir = os.path.join(os.path.dirname(output_path), f"{base_name}_files")
        
        # Make sure the directory exists
        os.makedirs(images_dir, exist_ok=True)
        
        # Also need to update the markdown file to point to the new location
        body_with_updated_paths = body
        
        for filename, data in resources['outputs'].items():
            # Save the resource to the subfolder
            resource_path = os.path.join(images_dir, filename)
            with open(resource_path, 'wb') as f:
                f.write(data)
            
            # Update the references in the markdown
            old_path = filename
            new_path = f"{base_name}_files/{filename}"
            body_with_updated_paths = body_with_updated_paths.replace(old_path, new_path)
        
        # Write the updated markdown content
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(body_with_updated_paths)
    
    return output_path

if __name__ == "__main__":
    # Replace with your notebook path
    notebook_path = 'README.ipynb'
    
    # Convert the notebook
    output_file = convert_notebook_to_markdown(notebook_path)
    print(f"Notebook converted successfully to {output_file}")