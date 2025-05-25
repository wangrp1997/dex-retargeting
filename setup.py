from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'dex_retargeting'

setup(
    name=package_name,
    version='0.5.0',
    packages=find_packages(exclude=['test']),
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
        'dex_retargeting',
    ],
    zip_safe=True,
    maintainer='rw',
    maintainer_email='renpengwang@eqq.com',
    description='ROS2 package for dex hand retargeting',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'retargeting_node = dex_retargeting.retargeting:main',
        ],
    },
)
