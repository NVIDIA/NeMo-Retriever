import logging
import uuid
from json import JSONEncoder
from nemo_retriever.relational_db.population.graph.model.reserved_words import (
    Labels,
    label_to_type,
)

logger = logging.getLogger(__name__)


class Node:
    def __init__(
        self,
        name,
        label,
        props,
        existing_id=None,
        match_props=None,
        override_existing_props=None,
    ):
        self.name = name
        self.label = label
        self.props = props
        if existing_id is None:
            self.id = str(uuid.uuid4())
        else:
            self.id = str(existing_id)

        self.props.update({"id": str(self.id)})
        if "type" not in self.props:
            self.props.update({"type": label_to_type(label)})

        self.match_props = match_props
        if match_props is None:
            self.match_props = {"id": str(self.id)}

        self.override_existing_props = override_existing_props if override_existing_props else None

    def change_label(self, new_label):
        self.label = new_label

    def get_name(self):
        return self.name

    def get_label(self):
        return self.label

    def get_properties(self) -> dict:
        return self.props

    def get_match_props(self) -> dict:
        return self.match_props

    def get_override_existing_props(self) -> dict:
        return self.override_existing_props

    def set_override_existing_props(self, override_props: dict) -> dict:
        self.override_existing_props = override_props

    def add_property(self, prop_name, prop_val):
        self.props.update({prop_name: prop_val})

    def add_properties(self, properties: dict):
        self.props.update(properties)

    def get_props_str(self):
        if self.props is None:
            return "{}"
        return str(self.props)

    def get_id(self):
        return self.id

    def replace_id(self, id):
        self.id = id
        self.props.update({"id": str(self.id)})
        if "id" in self.match_props:
            self.match_props.update({"id": str(self.id)})

    def pop_property(self, key):
        val = None
        if key in self.props.keys():
            val = self.props.pop(key)
        return val

    def restore_property(self, key, val):
        if val is not None:
            self.props.update({key: val})

    def __eq__(self, other):
        if isinstance(other, Node):
            # compare without uuid
            equal_result = (
                self.name == other.name and self.label == other.label and self.match_props == other.match_props
            )
            return equal_result

    def __str__(self):
        self_id = self.pop_property("id")
        str_val = "Node: {name: " + self.name + "; label: " + self.label + "; properties: " + str(self.props) + "}"
        self.restore_property("id", self_id)

        return str_val

    def __repr__(self):
        self_id = self.pop_property("id")
        repr_val = "Node: {name: " + self.name + "; label: " + self.label + "; properties: " + str(self.props) + "}"
        self.restore_property("id", self_id)

        return repr_val

    def __hash__(self):
        hash_val = hash((self.name, self.label, str(self.match_props)))
        return hash_val


# subclass JSONEncoder
class NodeEncoder(JSONEncoder):
    def default(self, o):
        props_copy = o.props.copy()
        if "id" in props_copy.keys():
            props_copy.pop("id")

        if o.label == Labels.SQL:
            props_copy.pop("name")
            dict_for_json = {"label": o.label, "properties": props_copy}
        else:
            dict_for_json = {"name": o.name, "label": o.label, "properties": props_copy}
        return dict_for_json
