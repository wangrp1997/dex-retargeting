from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dex_retargeting'

# 获取虚拟环境的 Python 解释器路径
venv_python = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'venv', 'bin', 'python3')

setup(
    name='dex_retargeting',
    version='0.5.0',
    packages=find_packages(where='.', include=['dex_retargeting', 'dex_retargeting.src', 'dex_retargeting.src.dex_retargeting']),
    package_dir={'': '.'},
    package_data={
        'dex_retargeting': [
            'src/dex_retargeting/configs/offline/*.yml',
            'src/dex_retargeting/configs/teleop/*.yml',
        ],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'assets', 'robots', 'hands'), 
         glob(os.path.join('dex_retargeting', 'assets', 'robots', 'hands', '*', '*.urdf'))),
        (os.path.join('share', package_name, 'assets', 'robots', 'hands', 'botyard_hand', 'meshes'),
         glob(os.path.join('dex_retargeting', 'assets', 'robots', 'hands', 'botyard_hand', 'meshes', '*.STL'))),
    ],
    install_requires=[
        'setuptools',
        'torch',
        'opencv-python',
        'loguru',
        'sapien==3.0.0b0',
        'anytree',
        'six',
        'pytransform3d',
    ],
    zip_safe=True,
    maintainer='rw',
    maintainer_email='renpengwang@eqq.com',
    description='ROS2 package for dex hand retargeting',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'retargeting_node = dex_retargeting.retargeting_oak:main',
        ],
    },
    python_requires='>=3.7,<3.13',
    options={
        'build_scripts': {
            'executable': venv_python,
        },
    },
)
