import json
import os

from app.settings import settings

from jinja2 import Environment, FileSystemLoader


def render_template(template, template_dir, **keys):
    with open(f"{template_dir}/{template}.jinja") as f:
        template_str = f.read()

    template = Environment(
        loader=FileSystemLoader(template_dir)
    ).from_string(template_str)
    instruction = template.render(**keys)
    return instruction


class Lens:
    def __init__(self, lens_type):
        self.lens_type = lens_type
        self.template_dir = os.path.join(settings.LENS_DIR, lens_type)
        self.function = self.get_function()
        self.system_prompt = self.get_system_template()
        self.config = self.get_config()
        self.input_fields = self.config["input_fields"]

    def get_system_template(self):
        return render_template("system", self.template_dir)

    def prompt_template(self, *args):
        if len(args) != len(self.input_fields):
            raise ValueError(f"Missing one or more required fields {self.input_fields} for lens {self.lens_type}")

        kwargs = dict(zip(self.input_fields, args))
        return render_template("prompt", self.template_dir, **kwargs)

    def get_function(self):
        with open(f"{self.template_dir}/function.json") as f:
            functions = json.load(f)
        return functions

    def get_config(self):
        with open(f"{self.template_dir}/config.json") as f:
            config = json.load(f)
        return config

    def labels(self):
        return self.function["parameters"]["required"]

    def score_labels(self):
        return [l for l in self.labels() if self.function["parameters"]["properties"][l]["type"] in ["integer", "float", "number"]]

    def rationale_labels(self):
        return [l for l in self.labels() if self.function["parameters"]["properties"][l]["type"] == "string"]

    def rater_type(self):
        return self.config["type"]

