from pathlib import Path
import re
from setuptools import setup, find_packages

this_directory = Path(__file__).parent
readme_path = this_directory / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

init_path = this_directory / "cevell_tunnel" / "__init__.py"
version_match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', init_path.read_text(encoding="utf-8"), re.M)
version = version_match.group(1) if version_match else "0.2.0"

setup(
    name="cevell-tunnel",
    version=version,
    description="Agnostic Confidential Translation Layer and Transport Tunnel for Hardware-Attested CVMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Cevell",
    author_email="contact@mail.cevell.com",
    url="https://github.com/cevell/cevell-tunnel",
    project_urls={
        "Homepage": "https://github.com/cevell/cevell-tunnel",
        "Repository": "https://github.com/cevell/cevell-tunnel",
        "Issues": "https://github.com/cevell/cevell-tunnel/issues",
    },
    license="Apache-2.0",
    packages=find_packages(exclude=["tests*", "tests"]),
    package_data={
        "cevell_tunnel": ["py.typed"],
    },
    install_requires=[
        "cryptography>=38.0.0",
    ],
    extras_require={
        "transport": ["httpx>=0.24.0"],
    },
    entry_points={
        "console_scripts": [
            "cevell-tunnel=cevell_tunnel.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Security :: Cryptography",
        "Topic :: Internet :: Proxy Servers",
    ],
    python_requires=">=3.8",
)
