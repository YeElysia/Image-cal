from types import SimpleNamespace
from typing import Dict, Any, Union, Optional, Iterator, Tuple
from pathlib import Path
import logging
from . import YAML

# 设置日志记录器
LOGGER = logging.getLogger(__name__)

# 默认配置文件路径
DEFAULT_CFG_PATH = "default.yaml"

class IterableNamespace(SimpleNamespace):
    """
    可迭代的SimpleNamespace类，提供增强的属性访问和迭代功能。

    这个类扩展了SimpleNamespace类，添加了迭代、字符串表示和属性访问的额外方法。
    它被设计为一个方便的容器，用于存储和访问配置参数。

    Methods:
        __iter__: 返回命名空间属性的键值对迭代器。
        __str__: 返回对象的人类可读字符串表示。
        __getattr__: 提供带有帮助信息的自定义属性访问错误消息。
        get: 获取指定键的值，如果键不存在则返回默认值。

    Examples:
        >>> cfg = IterableNamespace(a=1, b=2, c=3)
        >>> for k, v in cfg:
        ...     print(f"{k}: {v}")
        a: 1
        b: 2
        c: 3
        >>> print(cfg)
        a=1
        b=2
        c=3
        >>> cfg.get("b")
        2
        >>> cfg.get("d", "default")
        'default'

    Note:
        与标准字典相比，这个类特别适合以更易访问和可迭代的格式存储配置参数。
    """

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        """返回命名空间属性的键值对迭代器。"""
        return iter(vars(self).items())

    def __str__(self) -> str:
        """返回对象的人类可读字符串表示。"""
        return "\n".join(f"{k}={v}" for k, v in vars(self).items())

    def __getattr__(self, attr: str) -> Any:
        """
        自定义属性访问错误消息，提供帮助信息。

        Args:
            attr (str): 尝试访问的属性名。

        Raises:
            AttributeError: 当属性不存在时，提供详细的错误信息。
        """
        name = self.__class__.__name__
        raise AttributeError(
            f"""
            '{name}' 对象没有属性 '{attr}'。
            """
        )

    def get(self, key: str, default: Any = None) -> Any:
        """
        如果指定的键存在则返回其值；否则返回默认值。

        Args:
            key (str): 要获取的属性名。
            default (Any): 如果键不存在时返回的默认值。

        Returns:
            Any: 属性的值或默认值。
        """
        return getattr(self, key, default)

def check_dict_alignment(cfg: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """检查配置字典的对齐情况"""
    for k, v in overrides.items():
        if k in cfg and type(v) != type(cfg[k]):
            LOGGER.warning(f"类型不匹配: {k}={v} ({type(v)}) vs {k}={cfg[k]} ({type(cfg[k])})")

def check_cfg(cfg: Dict[str, Any]) -> None:
    """检查配置的有效性"""
    # 基本类型检查
    for k, v in cfg.items():
        if not isinstance(v, (str, int, float, bool, list, dict, type(None))):
            LOGGER.warning(f"不支持的配置类型: {k}={v} ({type(v)})")

def cfg2dict(cfg: Union[str, Path, Dict[str, Any], SimpleNamespace]) -> Dict[str, Any]:
    """
    将配置对象转换为字典。

    Args:
        cfg (Union[str, Path, Dict[str, Any], SimpleNamespace]): 要转换的配置对象。可以是文件路径、
            字符串、字典或SimpleNamespace对象。

    Returns:
        Dict[str, Any]: 字典格式的配置对象。

    Examples:
        将YAML文件路径转换为字典：
        >>> config_dict = cfg2dict("config.yaml")

        将SimpleNamespace转换为字典：
        >>> from types import SimpleNamespace
        >>> config_sn = SimpleNamespace(param1="value1", param2="value2")
        >>> config_dict = cfg2dict(config_sn)

        直接传递已存在的字典：
        >>> config_dict = cfg2dict({"param1": "value1", "param2": "value2"})

    Note:
        - 如果cfg是路径或字符串，它会被作为YAML加载并转换为字典
        - 如果cfg是SimpleNamespace对象，它会被使用vars()转换为字典
        - 如果cfg已经是字典，它会原样返回
    """
    if isinstance(cfg, (str, Path)):
        cfg = YAML.load(cfg)  # 加载字典
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # 转换为字典
    return cfg


def get_cfg(cfg: Union[str, Path, Dict[str, Any], SimpleNamespace] = "default.yaml", 
           overrides: Optional[Dict[str, Any]] = None) -> IterableNamespace:
    """
    从文件或字典加载并合并配置数据，支持可选的覆盖值。

    参数：
        cfg (Union[str, Path, Dict[str, Any], SimpleNamespace]): 配置数据源。可以是文件路径、字典或
            SimpleNamespace对象。
        overrides (Optional[Dict[str, Any]]): 包含要覆盖基础配置的键值对的字典。

    返回：
        IterableSimpleNamespace: 包含合并后配置参数的命名空间。

    示例：
        >>> from utils.cfg import get_cfg
        >>> config = get_cfg()  # 加载默认配置
        >>> config_with_overrides = get_cfg("path/to/config.yaml", overrides={"epochs": 50, "batch_size": 16})

    注意：
        - 如果同时提供了`cfg`和`overrides`，`overrides`中的值将优先
        - 特殊处理确保配置的对齐和正确性，例如将数字类型的`project`和`name`转换为字符串
        - 函数对配置数据执行类型和值检查
    """
    cfg = cfg2dict(cfg)

    # 合并覆盖值
    if overrides:
        overrides = cfg2dict(overrides)
        if "save_dir" not in cfg:
            overrides.pop("save_dir", None)  # 忽略特殊的覆盖键
        check_dict_alignment(cfg, overrides)
        cfg = {**cfg, **overrides}  # 合并cfg和overrides字典（优先使用overrides）

    # 对数字类型的project/name进行特殊处理
    for k in "project", "name":
        if k in cfg and isinstance(cfg[k], (int, float)):
            cfg[k] = str(cfg[k])
    if cfg.get("name") == "model":  # 将model赋值给'name'参数
        cfg["name"] = str(cfg.get("model", "")).split(".")[0]
        LOGGER.warning(f"'name=model'自动更新为'name={cfg['name']}'")

    # 类型和值检查
    check_cfg(cfg)

    # 返回实例
    return IterableNamespace(**cfg)

if __name__ == "__main__":
    cfg = get_cfg("/home/yee/elysiam/python/RM-cal/cfg/default.yaml")
    print(cfg)