
echo "Creating lpsds installation packages"
python -m build

echo "Submitting newly packages to pypi"
python -m twine upload --repository testpypi dist/*

echo "Submission completed!"
