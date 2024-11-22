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
    
    Methods:
    --------
    load_template(template_name: str) -> jinja2.Template:
        Loads a Jinja2 template by name.
        
    render_prompt(template_name: str, variables: dict) -> str:
        Renders the specified template with provided variables.
    
    set_template_dir(template_dir: str) -> None:
        Sets a new directory for loading templates.
    """
    
    def __init__(self):
        """
        Initializes the PromptManager with a directory for Jinja2 templates.

        Parameters:
        -----------
        template_dir : str
            Path to the directory containing Jinja2 templates from config file.
        """
        self.template_dir = conf.template_path

        if not os.path.isdir(self.template_dir):
            raise ValueError(f"Template directory {self.template_dir} does not exist.")
        
        self.env = Environment(loader=FileSystemLoader(self.template_dir))
    
    def load_template(self, template_name: str):
        """
        Loads a Jinja2 template by name.

        Parameters:
        -----------
        template_name : str
            The name of the template to load.

        Returns:
        --------
        template : jinja2.Template
            The loaded Jinja2 template.
        
        Raises:
        -------
        TemplateNotFound:
            If the template is not found in the directory.
        """
        try:
            template = self.env.get_template(template_name)
            return template
        except TemplateNotFound:
            raise FileNotFoundError(f"Template {template_name} not found in {self.template_dir}")
    
    def render_prompt(self, template_name: str, variables: dict) -> str:
        """
        Renders a prompt from a template with given variables.

        Parameters:
        -----------
        template_name : str
            The name of the template to render.
        variables : dict
            A dictionary of variables to substitute into the template.

        Returns:
        --------
        rendered_prompt : str
            The rendered prompt as a string.
        """
        template = self.load_template(template_name)
        return template.render(variables)
    
    def set_template_dir(self, template_dir: str) -> None:
        """
        Updates the template directory to a new path.

        Parameters:
        -----------
        template_dir : str
            The new directory to set for loading templates.
        """
        if not os.path.isdir(template_dir):
            raise ValueError(f"Template directory {template_dir} does not exist.")
        
        self.template_dir = template_dir
        self.env.loader = FileSystemLoader(self.template_dir)
    
    def list_templates(self):
        """
        Lists all available templates in the current template directory.

        Returns:
        --------
        list of str:
            List of template file names in the template directory.
        """
        return [f for f in os.listdir(self.template_dir) if f.endswith(".j2")]