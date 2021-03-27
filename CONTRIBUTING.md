# Contributing  to biobankAccelerometerAnalysis

*This document is based on [CONTRIBUTING.md](https://github.com/spring-projects/spring-framework/blob/master/CONTRIBUTING.md) of the popular Spring Framework repository*

First off, thank you for taking the time to contribute! :+1: :tada: 

### Table of Contents

* [How to Contribute](#how-to-contribute)
    * [Create an Issue](#create-an-issue)
    * [Submit a Pull Request](#submit-a-pull-request)
    * [Participate in Reviews](#participate-in-reviews)
    * [Keep Calm and Drink Tea](#keep-calm-and-drink-tea)
* [Source Code Style](#source-code-style)
* [Fork and Pull Request Workflow](#fork-and-pull-request-workflow)
    * [Setup](#workflow)
    * [Workflow](#workflow)
    * [Tips](#tips)
* [References](#references)

### How to Contribute

#### Create an Issue

Reporting an issue or making a feature request is a great way to contribute.
Your feedback and the conversations that result from it provide a continuous
flow of ideas. However, before creating a ticket, please take the time to
check for [existing
issues](https://github.com/activityMonitoring/biobankAccelerometerAnalysis/issues)
first.

#### Submit a Pull Request

Here we follow the [Fork & Pull Request Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow), so make sure to understand how that works, or [see at the end](#fork-and-pull-request-workflow) for a brief summary.

1. Should you create an issue first? No, just create the pull request and use
the description to provide context and motivation, as you would for an issue.
But if you still want to start a discussion first or have already created an issue, then reference the issue in your PR.

1. Choose the granularity of your commits consciously and squash commits that
represent multiple edits or corrections of the same logical change. See
[Rewriting History section of Pro
Git](https://git-scm.com/book/en/Git-Tools-Rewriting-History) for an overview
of streamlining the commit history.

1. Format commit messages using 55 characters for the subject line, 72
characters per line for the description, followed by the issue fixed. Use the
subject line to answer what the commit does. Use imperative form: "Fix ABC"
and not "Fixed ABC" or "Fixes ABC" (Tip: try to finish the sentence: "If
merged, this commit will..."). See the [Commit Guidelines section of Pro
Git](https://git-scm.com/book/en/Distributed-Git-Contributing-to-a-Project#Commit-Guidelines)
for best practices around commit messages.

1. When your PR is first submitted, or every time it is modified, automatic
checks will be performed in the background. The checks include running the
code with your proposed changes to process a sample accelerometer file and
verifying that the processing result hasn't changed. Fix any failed checks unless
they are expected (e.g. your bugfix correctly changes the processing result)
and discuss this in the PR ticket.

If accepted, your contribution may be heavily modified as needed prior to
merging. You will likely retain author attribution for your Git commits
granted that the bulk of your changes remain intact. You may also be asked to
rework the submission.

#### Participate in Reviews

Helping to review pull requests is another great way to contribute. Your
feedback can help to shape the implementation of new features. When reviewing
pull requests, however, please refrain from approving or rejecting a PR
unless you are a maintainer.

#### Keep Calm and Drink Tea
We are a team of researchers first and developers second, so please be patient if your issue or PR is taking long to be addressed. Just sit and relax!

### Source Code Style

Our codebase contains a mix of Java and Python. As you may know, Java follows
camelCase naming convention whereas [Python follows
snake_case](https://www.python.org/dev/peps/pep-0008/#function-and-variable-names)
(python is a type of snake after all), but you may have noticed that we
use camelCase for Python too. This is a spillover from Java as the codebase
was mostly Java-based in the beginning (and still is for many important parts). Finally, we recommend that you use a
Python linter (e.g. [flake8](https://flake8.pycqa.org/en/latest/)) -- this will
help you follow standard [Python coding style](https://www.python.org/dev/peps/pep-0008/).

### Fork and Pull Request Workflow

There are a number of git workflows for collaboration -- see this [tutorial](https://www.atlassian.com/git/tutorials/comparing-workflows) for a nice discussion. Here we use the [Fork and Pull Request Workflow](https://www.atlassian.com/git/tutorials/comparing-workflows), or simply Forking workflow.

#### Setup

1. Fork the repository. A fork is just a server-side clone of the repository to your own account, and which is managed by the server (e.g. Github, Bitbucket). From git's point of view, there is no difference between forking and cloning.

1. Clone the fork to your own local computer. Note that you
clone your fork and *not* the original repository. At this stage, there are three repository copies that matter to you: the original repository, your server-side clone (fork), and your local clone. You can do *whatever you want* with the last two.

1. When you cloned your fork to your local computer, your clone automatically kept a reference to your fork, usually labelled as "origin" (you can check this with `git remote -v`). Now add a reference to the original repository, labelled as "upstream": 

    ```bash
    # In general: git remote add my_new_remote https://github.com/username/reponame.git
    git remote add upstream https://github.com/activityMonitoring/biobankAccelerometerAnalysis.git
    ```

    This is so that we can keep track of changes happening in the original repository and pull/merge them to your local clone. You will only be able to pull changes but not push any of your own (unless you are a maintainer with write permissions); instead, you can request the maintainers to pull your changes (a pull request).
    You will be able to pull/push from/to your fork just fine though.

#### Workflow

Suppose you want to work on a bugfix or a new feature. In short, the workflow consists of doing all the work in a new branch, then pushing this branch to your remote fork, and finally using the server's functionality to ask the maintainers of the original repository to pull your changes &mdash; a pull request (PR).

1. First off, make sure your local master branch is up-to-date &mdash; you want to start developing from the latest version of the code.

    ```bash
    # What's new in upstream?
    git fetch upstream
    # Switch to your master branch
    git checkout master
    # Merge whatever's new from upstream master
    git rebase upstream/master
    ```

1. From master, create the new branch that will be used for the intended work. Use something descriptive for the branch name such as `fix-memory-error` as this shows up in the history once it is merged.
Then, push the branch to your remote fork.

    ```bash
    # Switch to master -- we want to branch out from here
    git checkout master
    # Create the new branch
    git branch fix-memory-error
    # Push the new branch to your fork (origin)
    # You need -u flag only for newly created branches
    git push -u origin fix-memory-error
    ```

1. Now the fun part: Work on your implementation as you would normally do within the new branch.

    ```bash
    # Switch to new branch
    git checkout fix-memory-error
    # ... do some work ...
    git commit -m "fix some stuff"
    # ... some more work ...
    git commit -m "fix some other stuff"
    # Push the changes to your fork
    git push origin fix-memory-error
    ```

1. Once you are done, some time may have passed since the beginning of your work. Before moving on, it is best to check if new changes occurred in upstream master and resolve any conflict. If you skip this step, your PR may complain of a merge conflict if there were indeed changes, after which you will have to resolve it anyway. Also, use `rebase` instead of `merge` to resolve the conflict for a cleaner history. Make sure you understand the difference between [rebasing and merging](https://www.atlassian.com/git/tutorials/merging-vs-rebasing).

    ```bash
    # The first three commands are exactly the same as Step 1
    # We're syncing our local master with upstream master

    git fetch upstream
    git checkout master
    git rebase upstream/master

    # Now sync your working branch with your master branch

    # Switch back to working branch
    git checkout fix-memory-error
    # Merge what's new from master
    # You may have to resolve any merge conflict
    git rebase master

    # Push changes again to your fork
    # You may need to do a forced push here:
    # git push origin fix-memory-error -f
    git push origin fix-memory-error

    ```

1. Now go to your server account (e.g. Github) and choose the option to submit a PR to merge your work into the original repository. Your PR will show up in the original repository and be notified to the maintainers for review.

1. If changes are requested during the review, just implement them locally in your working branch, then push them to your remote fork. The server will automatically update the PR to reflect the changes.

1. During the review process, the master branch of the original repository may change. If so, just repeat Step 4 as necessary.

1. A common scenario during a long review process is as follows: You wish to implement a new cool feature, but it depends on the bugfix that you just submitted which is still under review. The solution is to create a new branch off the bugfix branch:

    ```
    ---o---o master
           |
           o---o---o fix-memory-error (under review)
                   |
                   o---o---o cool-feature (working here)
    ```

    When a change is requested during the review, switch to `fix-memory-error` to make the change, then switch back to `cool-feature` and rebase it onto `fix-memory-error` to incorporate the change and continue working. When the PR is finally accepted and `fix-memory-error` merged into master, your `cool-feature` branch will already be up-to-date with master and ready for another PR.

#### Tips

The Fork and Pull Request Workflow may look complicated but you get used to it pretty quickly. Some tips that make it easier to remember:

1. Your local clone is your main workplace, while your remote fork is your communication port with the original repository. Your remote fork is where you make your changes visible to the maintainers so that they can review and pull them.

1. Any branch other than master should be short-lived, associated with a temporal piece of work and a PR submission (e.g. a bugfix, a new feature). This is reflected in the way Github and Bitbucket will prompt you to delete the branch once the PR has been accepted and merged. Note that other workflows may not follow this (for example, some repositories have a permanent "development branch"). 

1. Check for upstream changes regularly to incorporate them into your work as you go. Merge conflicts are hard enough, but it's easier if you deal with them in small chunks. It also helps to see what others have done so that you don't duplicate or undo any work. At the very least, try to keep your master branch up-to-date.

### References
- The [tutorials provided by Atlassian Bitbucket](https://www.atlassian.com/git/tutorials) are very nice. Some cherry-picked sections:
    - [Fork and pull request workflow](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow)
    - [Resolving merge conflicts](https://www.atlassian.com/git/tutorials/using-branches/merge-conflicts)
    - [Merging vs. Rebasing](https://www.atlassian.com/git/tutorials/merging-vs-rebasing)
- [More on rebasing](https://medium.com/singlestone/a-git-workflow-using-rebase-1b1210de83e5) (a Medium article, so may have a paywall)