"""
This module provides utilities for command line interface (CLI) parameter parsing and version information handling.
It includes functionality for parsing key-value pairs with type detection, version printing, and automatic type conversion.
The module is designed to work with the Click framework for creating command line applications.

Key features:
- Version information printing
- Key-value pair parsing with type hints
- Automatic type detection for CLI parameters
- Support for various data types including int, float, bool, str, none, and list
"""

import re
from typing import Dict, Any, Tuple, Optional

import click
from click.core import Context, Option

GLOBAL_CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


def print_version(module, ctx: Context, param: Option, value: bool) -> None:
    """
    Print version information of the CLI application.

    :param module: The module using this CLI.
    :type module: Any
    :param ctx: The Click context object.
    :type ctx: Context
    :param param: The parameter's metadata.
    :type param: Option
    :param value: The value of the current parameter.
    :type value: bool
    :return: None
    :rtype: None
    """
    _ = param
    if not value or ctx.resilient_parsing:
        return  # pragma: no cover

    click.echo(f'Module utils of {module}')
    ctx.exit()


def parse_key_value(ctx, param, value) -> Dict[str, Any]:
    """
    Parse key-value pairs from command line arguments with automatic type detection.

    Supported formats:
    - KEY=VALUE       # Auto-detect type (int, float, bool, None)
    - KEY:str=VALUE   # Force string type
    - KEY:int=VALUE   # Force integer type
    - KEY:float=VALUE # Force float type
    - KEY:bool=VALUE  # Force boolean type
    - KEY:none=VALUE  # Force None type (ignores VALUE)
    - KEY:list=1,2,3  # Parse as list

    :param ctx: The Click context object.
    :type ctx: Context
    :param param: The parameter's metadata.
    :type param: Option
    :param value: List of key-value strings to parse.
    :type value: List[str]
    :return: Dictionary containing parsed key-value pairs.
    :rtype: Dict[str, Any]
    :raises click.BadParameter: If the input format is invalid or type conversion fails.
    """
    result = {}
    if not value:
        return result

    for item in value:
        # Check for type specifier
        type_match = re.match(r'^([^=:]+)(?::([a-z]+))?=(.*)', item)
        if not type_match:
            raise click.BadParameter(f"Invalid format for {item}, expected KEY=VALUE or KEY:type=VALUE")

        key, type_hint, val = type_match.groups()
        key = key.strip()

        # Convert value based on type hint or auto-detection
        if type_hint:
            if type_hint == 'str':
                result[key] = val
            elif type_hint == 'int':
                try:
                    result[key] = int(val)
                except ValueError:
                    raise click.BadParameter(f"Cannot convert '{val}' to int for key '{key}'")
            elif type_hint == 'float':
                try:
                    result[key] = float(val)
                except ValueError:
                    raise click.BadParameter(f"Cannot convert '{val}' to float for key '{key}'")
            elif type_hint == 'bool':
                val = val.lower()
                if val in ('true', 'yes', 'y', '1'):
                    result[key] = True
                elif val in ('false', 'no', 'n', '0'):
                    result[key] = False
                else:
                    raise click.BadParameter(f"Cannot convert '{val}' to bool for key '{key}'")
            elif type_hint == 'none':
                result[key] = None
            elif type_hint == 'list':
                try:
                    elements = val.split(',')
                    parsed_elements = []
                    for element in elements:
                        parsed_elements.append(auto_detect_type(element))
                    result[key] = parsed_elements
                except Exception as e:
                    raise click.BadParameter(f"Error parsing list '{val}' for key '{key}': {str(e)}")
            else:
                raise click.BadParameter(f"Unknown type hint '{type_hint}' for key '{key}'")
        else:
            result[key] = auto_detect_type(val)

    return result


def auto_detect_type(value: str) -> Any:
    """
    Automatically detect and convert string value to appropriate type.

    :param value: The string value to convert.
    :type value: str
    :return: Converted value in appropriate type (int, float, bool, None, or str).
    :rtype: Any
    """
    value = value.strip()

    # Check for None
    if value.lower() == 'none':
        return None

    # Check for boolean values
    if value.lower() in ('true', 'yes', 'y'):
        return True
    if value.lower() in ('false', 'no', 'n'):
        return False

    # Check for numbers
    try:
        # Try parsing as integer
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)

        # Try parsing as float
        float_val = float(value)
        return float_val
    except ValueError:
        # If not a number, keep as string
        return value


def parse_tuple(ctx, param, value) -> Optional[Tuple[Any, ...]]:
    if value is not None:
        return tuple([auto_detect_type(v) for v in value.split(',')])
    else:
        return None
