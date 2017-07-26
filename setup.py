				

from setuptools	import setup, find_packages
setup(
	name='food_detector',
	version='1.0',
	author='Vishnu Vikash',
	authour_email='vishnuvikash@ou.edu',
	packages=find_packages(exclude=('tests', 'docs')),
	setup_requires=['pytest-runner'],
	tests_require=['pytest']				
)
