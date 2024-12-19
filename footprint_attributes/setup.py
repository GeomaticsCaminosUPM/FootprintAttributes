from setuptools import setup, find_packages

setup(
    name="footprint_attributes",
    version="0.1.0",
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Miguel Ureña Pliego",
    author_email="miguel.urena@upm.es",
    url="https://github.com/GeomaticsCaminosUPM/FootprintAttributes",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "geopandas>=0.12.0",
        "pandas>=1.3.0",
        "shapely>=1.8.0",
        "numpy>=1.21.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
