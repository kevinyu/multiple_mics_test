import copy
from typing import List, Union

import json


def fill_in_at(
        to_dicts: Union[List[dict], dict],
        key_path: List[str],
        set_value=None,
        delete: bool=False,
        force: bool=False,
        ):
    """Fill in a dict at all paths matching the given key_path

    Key paths are encoded a list of nested keys that do not necessarily have to
    exist already. A key ending in "[]" indicates a list and that the search will
    branch out to all subelements of that list (they must be dictionaries).

    Params
    ======
    to_dicts : List[dict] | dict
        A dict or list of dicts to update values in
    key_path : List[str]
        A list of keys to search down. A key ending in "[]" denotes a list and the search
        will continue in all elements of that list.
    set_value : object (default=None)
        The value to set at all paths matching key_path
    delete : bool (default=False)
        Delete the keys matching the key_path instead of updating a value
    force : bool (default=False)
        If setting value (not deleting), will overwrite existing values at the key_path.
        Otherwise, will only set values that do not already exist.

    Examples
    ========
    >>> example = {"a": 1, "b": {"b1": 2, "b2": 3}, "c": [{}, {"a": 4, "b": {"b2": 5}}]}
    >>> fill_in_at(example, ["a"], 6)  # Will not overwrite
    >>> fill_in_at(example, ["a"], 6, force=True)  # Will overwrite
    >>> fill_in_at(example, ["b", "b3"], 7)  # Creates a new key in example["b"]
    >>> fill_in_at(example, ["c[]", "a"], 1)  # Adds key to example["c"][0] but does not overwrite example["c"][1]["a"]
    >>> fill_in_at(example, ["c[]", "b"], delete=True)  # Deletes the element example["c"][1]["b"]
    """
    if key_path[-1].endswith("[]"):
        raise ValueError("Cannot fill in value of parts list ending in list: {}".format(key_path))

    if isinstance(to_dicts, dict):
        to_dicts = [to_dicts]

    curr_key, remaining_keys = key_path[0], key_path[1:]

    # On the last part, attempt to apply the write operation
    if not remaining_keys:
        for base_dict in to_dicts:
            if force or curr_key not in base_dict:
                base_dict[curr_key] = set_value
            elif delete and curr_key in base_dict:
                del base_dict[curr_key]
        return

    next_levels = []
    if curr_key.endswith("[]"):
        curr_key = curr_key[:-2]
        next_levels = []
        for base_dict in to_dicts:
            if curr_key not in base_dict:
                base_dict[curr_key] = []
            else:
                for val in base_dict.get(curr_key, []):
                    if not isinstance(val, dict):
                        raise ValueError("A subelement of key '{}[]' was not a dict".format(curr_key))
                    next_levels.append(val)
    else:
        for base_dict in to_dicts:
            # Create a subdict if it doesn't exist (don't do this when our goal is to delete something anyway)
            if not delete and curr_key not in base_dict:
                base_dict[curr_key] = {}
            next_levels.append(base_dict[curr_key])
    return fill_in_at(next_levels, remaining_keys, set_value, delete=delete, force=force)


def read_all_values(from_dicts: Union[List[dict], dict], key_path: List[str], include_missing=True):
    """Gets all values from dict matching a key_path

    Params
    ======
    from_dicts : List[dict] | dict
        A dict or list of dicts to read values from
    key_path : List[str]
        A list of keys to search down. A key ending in "[]" denotes a list and the search
        will continue in all elements of that list.
    include_missing : bool (default=True)
        By default, will include None when a value at the key_path has not been set. If
        include_missing is set to False, will omit values where the key does not exist
        (note that it will still include values that exist but are None)

    Examples
    ========
    >>> example = {"a": 1, "b": {"b1": 2, "b2": 3}, "c": [{}, {"a": 4, "b": {"b2": 5}}]}
    >>> read_all_values(example, ["b", "b1"])
    [2]
    >>> read_all_values(example, ["c[]", "a"])
    [None, 4]
    >>> read_all_values(example, ["c[]", "a"], include_missing=False)
    [4]
    """
    if key_path[-1].endswith("[]"):
        raise ValueError("Cannot read value of key_path list ending in list: {}".format(key_path))

    if isinstance(from_dicts, dict):
        from_dicts = [from_dicts]

    curr_key, remaining_keys = key_path[0], key_path[1:]

    if not remaining_keys:
        results = []
        for base_dict in from_dicts:
            if include_missing:
                results.append(base_dict.get(curr_key, None))
            elif curr_key in base_dict:
                results.append(base_dict[curr_key])
        return results

    next_levels = []
    if curr_key.endswith("[]"):
        curr_key = curr_key[:-2]
        for base_dict in from_dicts:
            if curr_key in base_dict:
                for val in base_dict.get(curr_key, []):
                    if not isinstance(val, dict):
                        raise ValueError("A subelement of key '{}[]' was not a dict".format(curr_key))
                    next_levels.append(val)
    else:
        for base_dict in from_dicts:
            if curr_key in base_dict:
                next_levels.append(base_dict[curr_key])

    return read_all_values(next_levels, remaining_keys)


class NestedConfig(dict):
    """Dict that allows accessing config file with "." notation, and propogating values up and down

    Attributes
    ==========
        default_values : tuple of tuples (key, default_value))
        inherited_values : tuple of tuples (base_key, inherit_key)
            values at inherit_key should inherit the base_key when it does not exist

    Methods
    =======
    __getitem__(key)
        Overrides default get by splitting the key by "."
    inherited()
        Creates a dictionary where inherited values are filled in
    deherited(include_missing=True)
        Simplifies the dictionary by finding inherited keys whose values are all
        the same, moving them up in hierarchy and then deleting them from the
        inherited locations
    read_all(key)
        Reads all elements matching the key
    """
    default_values = tuple()
    inherited_values = tuple()

    def _str2list(self, key):
        return key.split(".")

    def __init__(self, *args, **kwargs):
        """Initialize dictionary and fill in default values if they do not exist
        """
        super().__init__(*args, **kwargs)
        for key, value in self.default_values:
            try:
                self[key]
            except KeyError:
                fill_in_at(self, self._str2list(key), value)

    def __getitem__(self, key, default=None):
        """Get item by dot notation
        """
        key_path = self._str2list(key)
        val = self
        for part in key_path[:-1]:
            val = val.get(part, {})

        if key_path[-1] not in val:
            raise KeyError
        else:
            return val.get(key_path[-1])

    def inherited(self):
        """Creates a dictionary where inherited values are filled in"""
        copied = NestedConfig(copy.deepcopy(self))
        for base_key, nested_key in self.inherited_values:
            base_value = self[base_key]
            fill_in_at(copied, self._str2list(nested_key), base_value)
        return copied

    def deherited(self, include_missing=True):
        """Create a dictionary where shared inherited values are moved up to higher levels

        Simplifies the dictionary by finding inherited keys whose values are all
        the same, moving them up in hierarchy and then deleting them from the
        inherited locations
        """
        copied = self.inherited()
        for base_key, nested_key in self.inherited_values:
            values = set(read_all_values(copied, self._str2list(nested_key), include_missing=include_missing))
            if len(values) == 1:
                fill_in_at(copied, self._str2list(base_key), list(values)[0])
                fill_in_at(copied, self._str2list(nested_key), delete=True)
        return copied

    def read_all(self, key):
        return read_all_values(self, self._str2list(key), include_missing=True)

    @classmethod
    def read_yaml(cls, path):
        import yaml
        with open(path, "r") as yaml_file:
            config = yaml.load(yaml_file, Loader=yaml.FullLoader)
            return cls(config)

    def to_yaml(self, path):
        import yaml
        with open(path, "w") as yaml_file:
            yaml.dump(self, yaml_file, default_flow_style=False)

    @classmethod
    def read_json(cls, path):
        with open(path, "r") as json_file:
            config = json.load(json_file)
            return cls(config)

    def to_json(self, path):
        with open(path, "w") as json_file:
            json.dump(self, json_file, indent=4, sort_keys=True)


class RecordingConfig(NestedConfig):
    default_values = (
        ("gain", 0.0),
        ("device_name", "default"),
        ("synchronized", True),
        ("detect.threshold", 1000),
        ("save.base_dir", "."),
        ("save.subdirectories", ("{name}", "{date}", "{hour}")),
        ("save.filename_format", "{name}_{timestamp}"),
        ("collect.triggered", 1000),
        ("collect.min_file_duration", 0.5),
        ("collect.max_file_duration", 20.0),
        ("collect.buffer_duration", 0.2),
    )
    inherited_values = (
        ("gain", "streams[].gain"),
        ("collect.triggered", "streams[].collect.triggered"),
        ("collect.min_file_duration", "streams[].collect.min_file_duration"),
        ("collect.max_file_duration", "streams[].collect.max_file_duration"),
        ("collect.buffer_duration", "streams[].collect.buffer_duration"),
        ("detect.threshold", "streams[].detect.threshold"),
    )


class PlotConfig(dict):
    """I didn't want the dot-based defaults for this one"""
    default_values = (
        ("spectrogram.min_level", -10),
        ("spectrogram.max_level", 150),
        ("spectrogram.min_freq", 500),
        ("spectrogram.max_freq", 8000),
        ("spectrogram.cmap", "afmhot"),
        ("amplitude.show_max_only", 1),
        ("amplitude.y_min", -500),
        ("amplitude.y_max", 9999),
        ("amplitude.downsample", 50),
        ("window", 5.0),
        ("rate", 48000),
        ("chunk", 1024),
    )

    def __init__(self, *args, **kwargs):
        """Initialize dictionary and fill in default values if they do not exist
        """
        super().__init__(*args, **kwargs)
        for key, value in self.default_values:
            try:
                self[key]
            except KeyError:
                self[key] = value
