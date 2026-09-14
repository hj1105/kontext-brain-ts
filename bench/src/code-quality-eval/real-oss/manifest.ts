import type { RealOssTask } from "./contracts.js";

const blueprintNameNode = "domain:flask:blueprint-name";

/**
 * A real SWE-bench Verified task. The source patch is deliberately absent.
 * Only the upstream regression-test patch is retained, and it is applied by
 * the grader after the coding agent has exited.
 */
export const flaskBlueprintNameTask: RealOssTask = {
  instanceId: "pallets__flask-5014",
  taskId: "task:real-oss:pallets__flask-5014",
  codebaseId: "codebase:github:pallets/flask@7ee9ceb7",
  repository: "pallets/flask",
  repositoryUrl: "https://github.com/pallets/flask.git",
  license: "BSD-3-Clause",
  baseCommit: "7ee9ceb71e868944a46e1ff00b506772a53a4f1d",
  upstreamIssueUrl: "https://github.com/pallets/flask/issues/5010",
  upstreamPullRequestUrl: "https://github.com/pallets/flask/pull/5014",
  publicPrompt: `Require a non-empty name for Blueprints.

Things do not work correctly if a Blueprint is given an empty name (for
example, pallets/flask#4944). Raise ValueError when a Blueprint is constructed
with an empty name. Preserve existing Blueprint behavior otherwise.`,
  acceptanceStatement:
    "The upstream issue behavior is implemented and the Blueprint suite has no regressions.",
  nonGoals: ["Editing tests", "Changing unrelated Blueprint behavior", "Changing another module"],
  risk: "low",
  codeRoots: ["src"],
  allowedPaths: ["src/flask/blueprints.py"],
  targets: [
    {
      workItemId: "work-item:flask-blueprint-name-validation",
      plannedSymbolId: "planned-symbol:flask:Blueprint.__init__",
      relativePath: "src/flask/blueprints.py",
      qualifiedName: "Blueprint.__init__",
      symbolKind: "method",
      binding: "required",
      responsibility: "Validate the Blueprint name when a Blueprint is constructed.",
      ontologyNodeIds: [blueprintNameNode],
      capabilityId: "capability:flask-blueprint-construction",
    },
  ],
  sourceIntegrity: [
    {
      relativePath: "LICENSE.rst",
      sha256: "489a8e1108509ed98a37bb983e11e0f7e1d31f0bd8f99a79c8448e7ff37d07ea",
    },
    {
      relativePath: "src/flask/blueprints.py",
      sha256: "9d296c3a0dffffbafd59ef15290a6d4c2061a470f3ad2b00709388d35d1dddc0",
    },
    {
      relativePath: "docs/blueprints.rst",
      sha256: "0efbff2912d81e24df4e1d9400657a153cbbfe93c9d3035a6e51ae6ba2d34734",
    },
  ],
  environment: {
    pythonVersion: "3.11",
    packages: [
      "pytest==7.2.2",
      "asgiref==3.6.0",
      "attrs==22.2.0",
      "blinker==1.5",
      "exceptiongroup==1.1.0",
      "iniconfig==2.0.0",
      "packaging==23.0",
      "pluggy==1.0.0",
      "python-dotenv==1.0.0",
      "tomli==2.0.1",
      "Werkzeug==2.2.3",
      "Jinja2==3.1.2",
      "MarkupSafe==2.1.2",
      "itsdangerous==2.1.2",
      "click==8.1.3",
    ],
  },
  publicTest: {
    command: "./.venv/bin/python",
    args: ["-m", "pytest", "-q", "tests/test_blueprints.py"],
  },
  hiddenTest: {
    patch: `diff --git a/tests/test_blueprints.py b/tests/test_blueprints.py
--- a/tests/test_blueprints.py
+++ b/tests/test_blueprints.py
@@ -256,6 +256,11 @@ def test_dotted_name_not_allowed(app, client):
         flask.Blueprint("app.ui", __name__)
${" "}
${" "}
+def test_empty_name_not_allowed(app, client):
+    with pytest.raises(ValueError):
+        flask.Blueprint("", __name__)
+
+
 def test_dotted_names_from_app(app, client):
     test = flask.Blueprint("test", __name__)
${" "}
`,
    patchSha256: "e16f06b260b5169a49397e9d571b5af70317cd23792e1437232fabf718fe8871",
    runner: {
      kind: "pytest-selectors",
      command: "./.venv/bin/python",
      args: ["-m", "pytest", "-q"],
    },
    failToPass: ["tests/test_blueprints.py::test_empty_name_not_allowed"],
    passToPass: [
      "tests/test_blueprints.py::test_blueprint_specific_error_handling",
      "tests/test_blueprints.py::test_blueprint_specific_user_error_handling",
      "tests/test_blueprints.py::test_blueprint_app_error_handling",
      "tests/test_blueprints.py::test_blueprint_prefix_slash[-/-/]",
      "tests/test_blueprints.py::test_blueprint_prefix_slash[/--/]",
      "tests/test_blueprints.py::test_blueprint_prefix_slash[/-/-/]",
      "tests/test_blueprints.py::test_blueprint_prefix_slash[/foo--/foo]",
      "tests/test_blueprints.py::test_blueprint_prefix_slash[/foo/--/foo/]",
      "tests/test_blueprints.py::test_blueprint_prefix_slash[-/bar-/bar]",
      "tests/test_blueprints.py::test_blueprint_prefix_slash[/foo/-/bar-/foo/bar]",
      "tests/test_blueprints.py::test_blueprint_prefix_slash[/foo/-bar-/foo/bar]",
      "tests/test_blueprints.py::test_blueprint_prefix_slash[/foo-/bar-/foo/bar]",
      "tests/test_blueprints.py::test_blueprint_prefix_slash[/foo/-//bar-/foo/bar]",
      "tests/test_blueprints.py::test_blueprint_prefix_slash[/foo//-/bar-/foo/bar]",
      "tests/test_blueprints.py::test_blueprint_url_defaults",
      "tests/test_blueprints.py::test_blueprint_url_processors",
      "tests/test_blueprints.py::test_templates_and_static",
      "tests/test_blueprints.py::test_default_static_max_age",
      "tests/test_blueprints.py::test_templates_list",
      "tests/test_blueprints.py::test_dotted_name_not_allowed",
      "tests/test_blueprints.py::test_dotted_names_from_app",
      "tests/test_blueprints.py::test_empty_url_defaults",
      "tests/test_blueprints.py::test_route_decorator_custom_endpoint",
      "tests/test_blueprints.py::test_route_decorator_custom_endpoint_with_dots",
      "tests/test_blueprints.py::test_endpoint_decorator",
      "tests/test_blueprints.py::test_template_filter",
      "tests/test_blueprints.py::test_add_template_filter",
      "tests/test_blueprints.py::test_template_filter_with_name",
      "tests/test_blueprints.py::test_add_template_filter_with_name",
      "tests/test_blueprints.py::test_template_filter_with_template",
      "tests/test_blueprints.py::test_template_filter_after_route_with_template",
      "tests/test_blueprints.py::test_add_template_filter_with_template",
      "tests/test_blueprints.py::test_template_filter_with_name_and_template",
      "tests/test_blueprints.py::test_add_template_filter_with_name_and_template",
      "tests/test_blueprints.py::test_template_test",
      "tests/test_blueprints.py::test_add_template_test",
      "tests/test_blueprints.py::test_template_test_with_name",
      "tests/test_blueprints.py::test_add_template_test_with_name",
      "tests/test_blueprints.py::test_template_test_with_template",
      "tests/test_blueprints.py::test_template_test_after_route_with_template",
      "tests/test_blueprints.py::test_add_template_test_with_template",
      "tests/test_blueprints.py::test_template_test_with_name_and_template",
      "tests/test_blueprints.py::test_add_template_test_with_name_and_template",
      "tests/test_blueprints.py::test_context_processing",
      "tests/test_blueprints.py::test_template_global",
      "tests/test_blueprints.py::test_request_processing",
      "tests/test_blueprints.py::test_app_request_processing",
      "tests/test_blueprints.py::test_app_url_processors",
      "tests/test_blueprints.py::test_nested_blueprint",
      "tests/test_blueprints.py::test_nested_callback_order",
      "tests/test_blueprints.py::test_nesting_url_prefixes[/parent-/child-None-None]",
      "tests/test_blueprints.py::test_nesting_url_prefixes[/parent-None-None-/child]",
      "tests/test_blueprints.py::test_nesting_url_prefixes[None-None-/parent-/child]",
      "tests/test_blueprints.py::test_nesting_url_prefixes[/other-/something-/parent-/child]",
      "tests/test_blueprints.py::test_nesting_subdomains",
      "tests/test_blueprints.py::test_child_and_parent_subdomain",
      "tests/test_blueprints.py::test_unique_blueprint_names",
      "tests/test_blueprints.py::test_self_registration",
      "tests/test_blueprints.py::test_blueprint_renaming",
    ],
  },
  sourceDocuments: [
    {
      documentId: "github:pallets/flask:issue:5010",
      kind: "issue",
      title: "Require a non-empty name for Blueprints",
      body: `Things do not work correctly if a Blueprint is given an empty name (e.g. #4944).
It would be helpful if a ValueError was raised when trying to do that.`,
      sourceUrl: "https://github.com/pallets/flask/issues/5010",
      sourceSpan: "issue #5010 title and description",
      observedAt: "2023-03-04T00:09:15.000Z",
      ontologyNodeIds: [blueprintNameNode],
    },
    {
      documentId: "github:pallets/flask:issue:4944",
      kind: "issue",
      title: "Blueprint 404 handlers are not invoked for Blueprints with an empty name",
      body: `The docs suggest that, if a blueprint's view function calls abort(404),
the blueprint's errorhandler(404) will be called. However, that does not seem
to be the case in this reproduction:

from flask import Flask, Blueprint, abort

app = Flask(__name__)
blueprint = Blueprint("", __name__)

@blueprint.route("/")
def router():
    abort(404)

@blueprint.errorhandler(404)
def handler(e):
    return "it worked", 404

After registering the blueprint, a request to "/" returns the application's
default 404 response instead of "it worked".`,
      sourceUrl: "https://github.com/pallets/flask/issues/4944",
      sourceSpan: "issue #4944 description and reproduction",
      observedAt: "2023-01-13T00:18:42.000Z",
      ontologyNodeIds: [blueprintNameNode],
    },
    {
      documentId: "github:pallets/flask:pull:5014",
      kind: "pull_request",
      title: "Require a non empty name for blueprints",
      body: `State: merged
Merged at: 2023-03-11T16:34:56Z
Base commit: 7ee9ceb71e868944a46e1ff00b506772a53a4f1d

fixes #5010
ValueError is raised if a Blueprint is given an empty name`,
      sourceUrl: "https://github.com/pallets/flask/pull/5014",
      sourceSpan: "pull request #5014 metadata and description; source diff excluded",
      observedAt: "2023-03-11T16:34:56.000Z",
      ontologyNodeIds: [blueprintNameNode],
    },
    {
      documentId: "github:pallets/flask:docs:blueprints",
      kind: "documentation",
      title: "Modular Applications with Blueprints",
      body: `The basic concept of blueprints is that they record operations to
execute when registered on an application. Flask associates view functions
with blueprints when dispatching requests and generating URLs from one endpoint
to another.

When you bind a function with the help of the @simple_page.route decorator,
the blueprint will record the intention of registering the function on the
application when it is later registered. Additionally it will prefix the
endpoint of the function with the name of the blueprint which was given to the
Blueprint constructor. The blueprint's name does not modify the URL, only the
endpoint.`,
      sourceUrl:
        "https://github.com/pallets/flask/blob/7ee9ceb71e868944a46e1ff00b506772a53a4f1d/docs/blueprints.rst#L50-L82",
      sourceSpan: "docs/blueprints.rst:50-82 at base commit 7ee9ceb7",
      observedAt: "2023-03-11T16:23:08.000Z",
      ontologyNodeIds: [blueprintNameNode],
    },
  ],
  normativeRecords: [
    {
      kind: "decision",
      recordId: "decision:flask-blueprint-name-required",
      revisionId: "decision:flask-blueprint-name-required@pr-5014-merged",
      statement:
        "Constructing a Blueprint with an empty name raises ValueError; existing behavior for non-empty names remains unchanged.",
      evidenceIds: ["github:pallets/flask:issue:5010", "github:pallets/flask:pull:5014"],
      ontologyNodeIds: [blueprintNameNode],
    },
    {
      kind: "domain_term",
      recordId: "domain-term:flask-blueprint-name",
      revisionId: "domain-term:flask-blueprint-name@7ee9ceb7",
      term: "Blueprint name",
      definition:
        "The name supplied to the Blueprint constructor that namespaces endpoint names; it does not change the URL.",
      avoid: ["URL prefix", "application name"],
      evidenceIds: ["github:pallets/flask:docs:blueprints"],
      ontologyNodeIds: [blueprintNameNode],
    },
    {
      kind: "invariant",
      recordId: "invariant:flask-blueprint-name-valid",
      revisionId: "invariant:flask-blueprint-name-valid@pr-5014-merged",
      statement:
        "A Blueprint name is non-empty and contains no dot; invalid names fail at construction with ValueError.",
      evidenceIds: [
        "github:pallets/flask:issue:5010",
        "github:pallets/flask:pull:5014",
        "github:pallets/flask:docs:blueprints",
      ],
      ontologyNodeIds: [blueprintNameNode],
    },
  ],
  limitations: [
    "The public issue states the requested behavior directly, so this task validates real-repository integration and governance discipline more than difficult knowledge retrieval.",
  ],
};
