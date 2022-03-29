Contributing
------------
|
| 1. **Fork** the repository. Here is a guide how to do this:
|   https://help.github.com/en/github/getting-started-with-github/fork-a-repo
| 2. Before you make changes, make sure you **synchronized** with the repository.
|   Here is a guide: https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/syncing-a-fork
| 4. **Edit** the python scripts (implement your improvements).
| 5. Edit the **version number** in all files by search and replace all, especially setup.py.
|   Therefore, somebody could identify, that he is using a package with your changes later on,
|   this will allow rechecking the results from the past, as every version is saved in git and can be restored.
| 6. **Build and install** by using the commands from the makefile:
| * conda install recommonmark
| * conda install twine
| * make build
| * make conda_build
| * make conda_install_local
| 7. **Restart** your integrated development environment (IDE)
|    eg. Spyder, VSCode, ... , so the IDE reloads all packages, and the changes are applied
|    (depends on the IDE, whether this is required).
| 8. **Test** your code.
| 9. Git add filenames, git commit -m "comment", **git push** to **your fork**.
| 10. If you successfully pushed to your repository, there should be a notification, when you log into
|    the web-version of github, to **pull a request** (this is the inquiry to make your changes applied
|    to the main repository, so all could profit from your changes). Describe the issue you where solving
|    and how and why it is solving the problem.
| 11. Hope that the maintainer approves your work and merges it.
