import sys
import os

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.insert(0, project_dir)

from jinja2 import Environment, FileSystemLoader, TemplateNotFound
import src.conf as conf


class PromptManager:
    """
    A class to manage and render prompts using Jinja2 templates.

    Attributes:
    -----------
    template_dir : str
        The directory containing the Jinja2 templates.
    env : jinja2.Environment
        The Jinja2 environment for loading and rendering templates.
    """
    
    def __init__(self):
        """
        Initializes the PromptManager with a directory for Jinja2 templates.
        """
        self.template_dir = conf.template_path

        if not os.path.isdir(self.template_dir):
            raise ValueError(f"Template directory {self.template_dir} does not exist.")
        
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
    
    def load_template(self, template_name: str):
        """
        Loads a Jinja2 template by name.
        """
        try:
            template = self.env.get_template(template_name)
            return template
        except TemplateNotFound:
            raise FileNotFoundError(f"Template {template_name} not found in {self.template_dir}")
    
    def render_prompt(self, template_name: str, variables: dict) -> str:
        """
        Renders a prompt from a template with given variables.
        """
        try:
            template = self.load_template(template_name)
            return template.render(variables)
        except Exception as e:
            raise ValueError(f"Error rendering template {template_name}: {e}")
    
    def set_template_dir(self, template_dir: str) -> None:
        """
        Updates the template directory to a new path.
        """
        if not os.path.isdir(template_dir):
            raise ValueError(f"Template directory {template_dir} does not exist.")
        
        self.template_dir = template_dir
        self.env.loader = FileSystemLoader(self.template_dir)
    
    def list_templates(self):
        """
        Lists all available templates in the current template directory.
        """
        return [f for f in os.listdir(self.template_dir) if f.endswith(".j2")]
