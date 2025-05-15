import re
from typing import Dict, Any

import click
from click.core import Context, Option

GLOBAL_CONTEXT_SETTINGS = dict(
    help_option_names=['-h', '--help']
)


def print_version(module, ctx: Context, param: Option, value: bool) -> None:
    """
    Print version information of cli
    :param module: current module using this cli.
    :param ctx: click context
    :param param: current parameter's metadata
    :param value: value of current parameter
    """
    _ = param
    if not value or ctx.resilient_parsing:
        return  # pragma: no cover

    click.echo(f'Module utils of {module}')
    ctx.exit()


# 假设GLOBAL_CONTEXT_SETTINGS已经在其他地方定义

def parse_key_value(ctx, param, value) -> Dict[str, Any]:
    """
    解析命令行参数中的键值对，自动检测值的类型

    支持的格式:
    - KEY=VALUE       # 自动检测类型 (int, float, bool, None)
    - KEY:str=VALUE   # 强制为字符串类型
    - KEY:int=VALUE   # 强制为整数类型
    - KEY:float=VALUE # 强制为浮点数类型
    - KEY:bool=VALUE  # 强制为布尔类型
    - KEY:none=VALUE  # 强制为None (忽略VALUE)
    - KEY:list=1,2,3  # 解析为列表
    """
    result = {}
    if not value:
        return result

    for item in value:
        # 检查是否有类型说明符
        type_match = re.match(r'^([^=:]+)(?::([a-z]+))?=(.*)$', item)
        if not type_match:
            raise click.BadParameter(f"Invalid format for {item}, expected KEY=VALUE or KEY:type=VALUE")

        key, type_hint, val = type_match.groups()
        key = key.strip()

        # 根据类型提示或自动检测来转换值
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
                    # 尝试将逗号分隔的值解析为列表，并自动检测每个元素的类型
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
            # 自动检测类型
            result[key] = auto_detect_type(val)

    return result


def auto_detect_type(value: str) -> Any:
    """自动检测并转换字符串值的类型"""
    value = value.strip()

    # 检查None
    if value.lower() == 'none':
        return None

    # 检查布尔值
    if value.lower() in ('true', 'yes', 'y'):
        return True
    if value.lower() in ('false', 'no', 'n'):
        return False

    # 检查数字
    try:
        # 尝试解析为整数
        if value.isdigit() or (value.startswith('-') and value[1:].isdigit()):
            return int(value)

        # 尝试解析为浮点数
        float_val = float(value)
        return float_val
    except ValueError:
        # 如果不是数字，则保留为字符串
        return value
