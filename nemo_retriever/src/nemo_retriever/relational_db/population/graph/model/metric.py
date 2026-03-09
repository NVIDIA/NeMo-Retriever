from shared.graph.model.node import Node
from .reserved_words import Labels, DataTypes  # , get_types_families
from .snippet import Snippet
import pandas as pd
from functools import reduce


class Metric:
    def __init__(
        self,
        name,
        id,
        account_id,
        description,
        formula,
        owner_id=None,
        user_id=None,
        owner_notes=None,
        recommended=False,
        simple_formula="",
        is_bi: bool = False,
    ):
        self.name = name
        self.id = id
        self.account_id = account_id
        self.description = description
        self.formula = formula
        self.simple_formula = simple_formula
        self.formula_edges: list[tuple[Node, Node]] = []
        self.definition_sql = []
        self.owner_id = owner_id
        self.user_id = user_id
        self.owner_notes = owner_notes
        self.recommended = recommended
        self.snippets_df: pd.DataFrame

        props = {  #'id': self.id,
            "name": self.name,
            "label": Labels.METRIC,
            "formula": self.formula,
            "simple_formula": self.simple_formula,
            "account_id": self.account_id,
            "recommended": self.recommended,
        }

        if self.description is not None:
            props["description"] = self.description
        if self.owner_id is not None:
            props["owner_id"] = self.owner_id
        if self.owner_notes is not None:
            props["owner_notes"] = self.owner_notes
        if self.user_id is not None:
            props["created_by"] = self.user_id
        if is_bi is not None:
            props["status"] = "generated"
        self.metric_node = Node(
            name=name, label=Labels.METRIC, props=props, existing_id=id
        )
        self.props = props

    def set_metric_snippets(self, snippets_df: pd.DataFrame):
        self.snippets_df = snippets_df

    def get_attr_id_from_term_attr_name(self, term_name: str, attr_name: str):
        attr_id = ""
        all_metric_snippets: list[Snippet] = (
            self.snippets_df.to_numpy().flatten().tolist()
        )
        for snippet in all_metric_snippets:
            if (
                snippet.attribute_name.lower() == attr_name.lower()
                and snippet.term_name.lower() == term_name.lower()
            ):
                attr_id = snippet.attribute_id
                break
        return attr_id

    def get_attr_data_types(self, attr_id: str):
        snippets_of_metric: list[dict[str, Snippet]] = self.snippets_df.to_dict(
            orient="records"
        )
        return reduce(
            lambda types, snippets_map: types
            + snippets_map[attr_id].data_item_data_type,
            snippets_of_metric,
            [],
        )

    def get_filtered_snippets_by_data_types(
        self, attr_id: str, valid_data_types: list[str]
    ) -> pd.DataFrame:
        snippets_list: list[dict[str, Snippet]] = self.snippets_df.to_dict(
            orient="records"
        )
        valid_snippets: list[dict[str, Snippet]] = []
        for snippets_combination in snippets_list:
            # snippet: Snippet = snippets_combination[attr_id]
            # valid_types_families = get_types_families(valid_data_types)
            # if set(get_types_families(snippet.data_item_data_type)).intersection(
            #     valid_types_families
            # ):
            valid_snippets.append(snippets_combination)
        return pd.DataFrame(valid_snippets, columns=snippets_combination.keys())

    def get_node_children_in_formula(self, node: Node):
        return [edges[1] for edges in self.formula_edges if edges[0] == node]

    ## an attribute could be represented by multiple different snippets
    ## so whenever there is a types compatibility restriction in the formula (operators/functions)-
    ## we want to filter out all the non compatible snippets combinations
    # for example an attribute has both string snippets and numeric snippets-
    # if the attribute is inside a sum() we would filter out the string snippets
    def filter_incompatible_combinations(
        self, op_or_func_node: Node, valid_data_types: list[DataTypes]
    ) -> list[DataTypes]:
        def find_attributes_leaves(current_root: Node, attributes: list[Node]):
            children = self.get_node_children_in_formula(current_root)
            is_leaf = len(children) == 0
            if is_leaf:
                leaf_is_attribute = current_root.label == Labels.ATTR
                if leaf_is_attribute:
                    attributes.append(current_root)
                return
            else:
                for child in children:
                    find_attributes_leaves(child, attributes)

        attributes_from_node: list[Node] = []
        find_attributes_leaves(op_or_func_node, attributes_from_node)
        for attribute in attributes_from_node:
            filtered_snippets_df = self.get_filtered_snippets_by_data_types(
                attribute.id, valid_data_types
            )
            self.set_metric_snippets(filtered_snippets_df)

    def get_edges(self):
        all_edges = []
        all_edges.extend(self.formula_edges)
        return all_edges

    def add_subselect(self, subselect_obj, id, section_type):
        self.subselects.update({id: subselect_obj})
        if section_type not in self.section_to_subselect.keys():
            self.section_to_subselect.update({section_type: []})
        self.section_to_subselect[section_type].append(subselect_obj)

    def add_edge(self, edge, section_type):
        self.add_edges([edge], section_type)

    def add_edges(self, edges, section_type):
        if section_type == "formula":
            self.formula_edges.extend(edges)
        else:
            raise Exception(
                "Unknown type " + section_type + " when adding the edges: " + edges
            )
