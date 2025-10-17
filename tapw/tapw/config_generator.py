import os
import shutil
import argparse
from importlib.resources import files
import yaml
import re
import subprocess

class PreservedScalarString(str): pass

def string_presenter(dumper, data):
    """Preserve multiline strings as block literals."""
    if len(data.splitlines()) > 1:  # check for multiline string
        return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='|')
    return dumper.represent_scalar('tag:yaml.org,2002:str', data)

def list_presenter(dumper, data):
    """Format short lists on a single line."""
    if len(data) > 0 and all(isinstance(item, (int, float, str)) for item in data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)
    return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=False)

yaml.add_representer(str, string_presenter)
yaml.add_representer(PreservedScalarString, string_presenter)
yaml.add_representer(list, list_presenter)

def get_default_config_path():
    """Get the path to the default config files in the package."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_yaml = os.path.join(current_dir, 'config.yaml')
    bands_yaml = os.path.join(current_dir, 'bands.yaml')
    # config_yaml = str(files('tapw').joinpath('config.yaml'))
    # bands_yaml = str(files('tapw').joinpath('bands.yaml'))
    return config_yaml, bands_yaml

def read_yaml_with_comments(file_path):
    """Read YAML file while preserving comments."""
    with open(file_path, 'r') as f:
        content = f.read()
    return content, yaml.safe_load(content)

def get_pwd(physical=False):
    """Get the current working directory using pwd command."""
    try:
        cmd = ['pwd', '-P'] if physical else ['pwd', '-L']
        # print("cmd is ", cmd)
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        # 如果 pwd 命令失败，回退到 Python 的方法
        if physical:
            return os.path.realpath(os.getcwd())
        return os.getcwd()

def get_real_path(output_dir, use_logical_path=True):
    """Get the real absolute path, handling relative paths correctly."""
    if output_dir == '.':
        # 使用 pwd 命令获取当前目录
        return get_pwd(not use_logical_path)
    else:
        # 如果是其他路径，先获取当前目录
        current_dir = get_pwd(not use_logical_path)
        
        # 如果是绝对路径，直接使用
        if os.path.isabs(output_dir):
            target_dir = output_dir
        else:
            # 相对于当前工作目录的路径
            target_dir = os.path.join(current_dir, output_dir)
        
        # 根据选项决定是否解析符号链接
        if use_logical_path:
            return os.path.abspath(target_dir)
        else:
            return os.path.realpath(target_dir)

def customize_config(config_content, config_data, output_dir, use_logical_path):
    """Customize configuration based on the current environment."""
    # 获取路径
    real_path = get_real_path(output_dir, use_logical_path)
    # logical_path = get_real_path(output_dir, True)
    # physical_path = get_real_path(output_dir, False)
    
    # print("Logical path (pwd -L):", logical_path)
    # print("Physical path (pwd -P):", physical_path)
    # print("Using path:", real_path)
    
    # 使用正则表达式更新 base_path，保留注释
    # config_content = re.sub(
    #     r'(base_path:).*',
    #     f'\\1 {real_path}',
    #     config_content
    # )
    
    # 使用正则表达式更新其他路径，保留注释
    config_content = re.sub(
        r'(kpath_in:).*',
        f'\\1 {real_path}/KPATH.in',
        config_content
    )
    config_content = re.sub(
        r'(kpath_out:).*',
        f'\\1 {real_path}/KPATH.out',
        config_content
    )
    config_content = re.sub(
        r'(output_dir:).*',
        f'\\1 {real_path}',
        config_content
    )
    
    # 更新 num_processes
    # config_content = re.sub(
    #     r'(num_processes:).*',
    #     f'\\1 {os.cpu_count() or 1}',
    #     config_content
    # )
    
    return config_content

def customize_bands(bands_content, bands_data,output_dir):
    """Customize bands configuration."""
    real_path = get_real_path(output_dir)
    # 更新标题
    bands_content = re.sub(
        r'(title:).*',
        '\\1 "Band Structure"',
        bands_content
    )
    
    # 使用正则表达式更新配置，保留注释
    bands_content = re.sub(
        r'(kpath_in:).*',
        f'\\1 {real_path}/KPATH.in',
        bands_content
    )
    bands_content = re.sub(
        r'(kpath_out:).*',
        f'\\1 {real_path}/KPATH.out',
        bands_content
    )
    
    return bands_content

def generate_config(output_dir='.', use_logical_path=True):
    """Generate configuration files in the specified directory."""
    config_yaml, bands_yaml = get_default_config_path()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取并自定义 config.yaml
    config_content, config_data = read_yaml_with_comments(config_yaml)
    config_content = customize_config(config_content, config_data, output_dir, use_logical_path)
    
    # 读取并自定义 bands.yaml
    bands_content, bands_data = read_yaml_with_comments(bands_yaml)
    bands_content = customize_bands(bands_content, bands_data,output_dir)
    
    # 保存自定义后的配置文件
    config_out = os.path.join(output_dir, 'config.yaml')
    bands_out = os.path.join(output_dir, 'bands.yaml')
    
    with open(config_out, 'w') as f:
        f.write(config_content)
    
    with open(bands_out, 'w') as f:
        f.write(bands_content)
    
    # 复制 KPATH 文件，但使用新的名字
    current_dir = os.path.dirname(os.path.abspath(__file__))
    kpath_in = os.path.join(current_dir, 'KPATH.in')
    shutil.copy2(kpath_in, os.path.join(output_dir, 'KPATH.in'))
    
    print(f"Configuration files generated in {output_dir}:")
    print(f"  - config.yaml (customized for current directory)")
    print(f"  - bands.yaml (with default settings)")
    print(f"  - KPATH.in")
    
    # 创建输出目录
    # os.makedirs(os.path.join(output_dir, 'output'), exist_ok=True)
    # print(f"  + output/ directory created")

def main():
    parser = argparse.ArgumentParser(description='Generate TAPW configuration files')
    parser.add_argument('-o', '--output', default='./output',
                      help='Output directory for configuration files (default: current directory)')
    parser.add_argument('-P', '--physical-path', action='store_true',
                      help='Use physical path (resolve symlinks) instead of logical path')
    
    args = parser.parse_args()
    generate_config(args.output, not args.physical_path)

if __name__ == '__main__':
    main() 