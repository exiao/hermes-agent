from pathlib import Path
import re

from ruamel.yaml import YAML


WORKFLOW_PATH = Path(__file__).parents[1] / ".github" / "workflows" / "claude-code-review.yml"


def _load_workflow() -> dict:
    yaml = YAML(typ="safe")
    return yaml.load(WORKFLOW_PATH.read_text(encoding="utf-8"))


def _find_step(job: dict, *, uses: str) -> dict:
    return next(step for step in job["steps"] if str(step.get("uses", "")).startswith(uses))


def test_fork_reviews_use_privilege_separation() -> None:
    workflow = _load_workflow()

    assert set(workflow["on"]) == {"pull_request_target"}
    assert workflow["concurrency"]["cancel-in-progress"] is True

    review = workflow["jobs"]["claude-review"]
    publish = workflow["jobs"]["publish-review"]

    assert "pull_request.user.type != 'Bot'" in review["if"]
    assert "pull_request.draft == false" in review["if"]
    assert review["permissions"] == {
        "contents": "read",
        "pull-requests": "read",
        "issues": "read",
    }
    assert publish["permissions"] == {"pull-requests": "write"}
    assert publish["needs"] == "claude-review"

    for job in (review, publish):
        for step in job["steps"]:
            if uses := step.get("uses"):
                assert re.search(r"@[0-9a-f]{40}$", uses)

    checkout = _find_step(review, uses="actions/checkout@")
    assert checkout["with"]["ref"] == "${{ github.event.pull_request.base.sha }}"
    assert checkout["with"]["persist-credentials"] is False

    capture = next(
        step
        for step in review["steps"]
        if step.get("name") == "Capture pull request as untrusted review input"
    )
    capture_script = capture["run"]
    assert "git fetch" in capture_script
    assert '"$HEAD_SHA"' in capture_script
    assert 'git diff --binary --no-ext-diff --no-textconv "$BASE_SHA" "$HEAD_SHA"' in capture_script
    assert "gh pr diff" not in capture_script

    claude = _find_step(review, uses="anthropics/claude-code-action@")
    inputs = claude["with"]
    assert inputs["github_token"] == "${{ github.token }}"
    assert inputs["allowed_non_write_users"] == "*"
    assert inputs.get("track_progress", False) is False
    assert "CLAUDE_CODE_OAUTH_TOKEN" in inputs["claude_code_oauth_token"]

    args = inputs["claude_args"]
    assert '--tools "Read,Glob,Grep"' in args
    assert "--max-budget-usd 2" in args
    for forbidden_tool in ("Bash", "Write", "Edit", "WebFetch", "WebSearch"):
        assert forbidden_tool in args.split("--disallowed-tools", 1)[1]

    assert "CLAUDE_CODE_OAUTH_TOKEN" not in str(publish)
    publisher = _find_step(publish, uses="actions/github-script@")
    assert publisher["env"] == {
        "EXPECTED_BASE_SHA": "${{ github.event.pull_request.base.sha }}",
        "EXPECTED_HEAD_SHA": "${{ github.event.pull_request.head.sha }}",
        "REVIEW_JSON": "${{ needs.claude-review.outputs.review }}",
    }
    publisher_script = publisher["with"]["script"]
    assert "pull.base.sha !== expectedBase" in publisher_script
    assert "pull.head.sha !== expectedHead" in publisher_script
    assert 'event: "COMMENT"' in publisher_script
