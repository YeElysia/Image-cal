from typing import Dict, Any, Union
from pathlib import Path
import re
import logging

# 设置日志记录器
LOGGER = logging.getLogger(__name__)

def colorstr(*input: str) -> str:
    """为字符串添加颜色 https://en.wikipedia.org/wiki/ANSI_escape_code"""
    if not input:
        return ''
    if len(input) > 1:
        *args, string = input
    else:
        args = ['blue', 'bold']
        string = input[0]  # type: ignore
    
    colors = {
        'black': '\033[30m',  # 基础颜色
        'red': '\033[31m',
        'green': '\033[32m',
        'yellow': '\033[33m',
        'blue': '\033[34m',
        'magenta': '\033[35m',
        'cyan': '\033[36m',
        'white': '\033[37m',
        'bright_black': '\033[90m',  # 亮色
        'bright_red': '\033[91m',
        'bright_green': '\033[92m',
        'bright_yellow': '\033[93m',
        'bright_blue': '\033[94m',
        'bright_magenta': '\033[95m',
        'bright_cyan': '\033[96m',
        'bright_white': '\033[97m',
        'end': '\033[0m',  # 其他
        'bold': '\033[1m',
        'underline': '\033[4m'}
    return ''.join(colors[x] for x in args) + f'{string}' + colors['end']


class YAML:
    """
    YAML工具类，用于高效的文件操作，具有自动C实现检测功能。

    该类使用PyYAML的最快可用实现（尽可能使用C实现）提供优化的YAML加载和保存操作。
    它实现了单例模式，支持延迟初始化，允许直接使用类方法而无需显式实例化。
    该类自动处理文件路径创建、验证和字符编码问题。

    实现通过以下方式优先考虑性能：
        - 自动选择可用的C实现加载器/转储器
        - 单例模式以重用同一实例
        - 延迟初始化以推迟导入成本
        - 处理问题YAML内容的回退机制

    属性：
        _instance: 内部单例实例存储
        yaml: PyYAML模块的引用
        SafeLoader: 最佳可用YAML加载器（如果可用则使用CSafeLoader）
        SafeDumper: 最佳可用YAML转储器（如果可用则使用CSafeDumper）

    示例：
        >>> data = YAML.load("config.yaml")
        >>> data["new_value"] = 123
        >>> YAML.save("updated_config.yaml", data)
        >>> YAML.print(data)
    """

    _instance = None

    @classmethod
    def _get_instance(cls):
        """首次使用时初始化单例实例"""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        """使用最佳YAML实现初始化（如果可用则使用C实现）"""
        import yaml

        self.yaml = yaml
        # 如果可用则使用C实现以获得更好的性能
        try:
            self.SafeLoader = yaml.CSafeLoader
            self.SafeDumper = yaml.CSafeDumper
        except (AttributeError, ImportError):
            self.SafeLoader = yaml.SafeLoader
            self.SafeDumper = yaml.SafeDumper

    @classmethod
    def save(cls, file: Union[str, Path] = "data.yaml", data: Dict[str, Any] | None = None, header: str = ""):
        """
        将Python对象保存为YAML文件。

        参数：
            file (str | Path): 保存YAML文件的路径
            data (dict | None): 要保存的字典或兼容对象
            header (str): 可选的文件开头字符串
        """
        instance = cls._get_instance()
        if data is None:
            data = {}

        # 如果需要则创建父目录
        file_path : Path = Path(file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # 将不可序列化的对象转换为字符串
        valid_types = int, float, str, bool, list, tuple, dict, type(None)
        for k, v in data.items():
            if not isinstance(v, valid_types):
                data[k] = str(v)

        # 写入YAML文件
        with open(file_path, "w", errors="ignore", encoding="utf-8") as f:
            if header:
                f.write(header)
            instance.yaml.dump(data, f, sort_keys=False, allow_unicode=True, Dumper=instance.SafeDumper)

    @classmethod
    def load(cls, file: Union[str, Path] = "data.yaml", append_filename: bool = False) -> Dict[str, Any]:
        """
        将YAML文件加载为Python对象，具有健壮的错误处理。

        参数：
            file (str | Path): YAML文件路径
            append_filename (bool): 是否在返回的字典中添加文件名

        返回：
            Dict[str, Any]: 加载的YAML内容
        """
        instance = cls._get_instance()
        assert str(file).endswith((".yaml", ".yml")), f"不是YAML文件: {file}"

        # 读取文件内容
        with open(file, errors="ignore", encoding="utf-8") as f:
            s = f.read()

        # 尝试加载YAML，对问题字符进行回退处理
        try:
            data: Dict[str, Any] = instance.yaml.load(s, Loader=instance.SafeLoader) or {}
        except Exception:
            # 移除问题字符并重试
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)
            data: Dict[str, Any] = instance.yaml.load(s, Loader=instance.SafeLoader) or {}

        # 检查用户错误输入的None字符串（在YAML中应该是'null'）
        if "None" in data.values():
            data = {k: None if v == "None" else v for k, v in data.items()}

        if append_filename:
            data["yaml_file"] = str(file)
        return data

    @classmethod
    def print(cls, yaml_file: Union[str, Path, Dict[str, Any]]) -> None:
        """
        将YAML文件或对象美观地打印到控制台。

        参数：
            yaml_file (Union[str, Path, Dict[str, Any]]): YAML文件路径或要打印的字典
        """
        instance = cls._get_instance()

        # 如果提供了路径则加载文件
        yaml_dict = cls.load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file

        # 在C实现中使用-1表示无限宽度
        dump = instance.yaml.dump(yaml_dict, sort_keys=False, allow_unicode=True, width=-1, Dumper=instance.SafeDumper)

        LOGGER.info(f"打印 '{colorstr('bold', 'black', str(yaml_file))}'\n\n{dump}")
