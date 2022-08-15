from distutils.core import setup

setup(
    name='PhenixFragments',
    version='0.1dev',
    packages=['frag',
              'frag.mol',
              "frag.featurization",
              "frag.fragmentation",
              "frag.labeling",
              "frag.graph",
              "frag.utils"],
    install_requires=[],
    license=open('LICENSE.txt').read(),
    long_description=open('README.md').read(),
)
