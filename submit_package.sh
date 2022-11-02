
echo "Creating lpsds installation packages"
python -m build

echo "Submitting newly packages to pypi"
python -m twine upload dist/*

echo "Submission completed!"
