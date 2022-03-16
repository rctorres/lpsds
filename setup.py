import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="rct-fast-ds",
    version="0.0.1",
    author="Rodrigo Coura Torres",
    author_email="torres.rc@gmail.com",
    description="Helper functions for data science projects.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rctorres/fast-ds",
    project_urls={
        "Bug Tracker": "https://github.com/rctorres/fast-ds/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)