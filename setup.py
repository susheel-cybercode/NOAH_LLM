from setuptools import setup, find_packages

setup(
    name="maya-ai",
    version="1.0.0",
    description="MAYA AI - The Ultimate AI Assistant",
    author="MAYA AI Team",
    packages=find_packages(),
    install_requires=[
        'flask>=2.0.0',
        'numpy>=1.20.0',
        'requests>=2.25.0',
        'python-dotenv>=0.19.0',
        'pillow>=8.0.0',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'maya-ai=maya_ai.main:main',
        ]
    }
)
