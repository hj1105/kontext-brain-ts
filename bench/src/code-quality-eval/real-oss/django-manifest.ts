import type { RealOssTask } from "./contracts.js";

const annotationDependencyNode = "domain:django:annotation-dependency";
const aggregationPlanningNode = "domain:django:aggregation-planning";
const queryPredicateNode = "domain:django:query-predicate";

const hiddenTestPatch = `diff --git a/tests/aggregation/tests.py b/tests/aggregation/tests.py
--- a/tests/aggregation/tests.py
+++ b/tests/aggregation/tests.py
@@ -34,6 +34,7 @@
     Cast,
     Coalesce,
     Greatest,
+    Lower,
     Now,
     Pi,
     TruncDate,
@@ -2084,3 +2085,41 @@ def test_exists_extra_where_with_aggregate(self):
             exists=Exists(Author.objects.extra(where=["1=0"])),
         )
         self.assertEqual(len(qs), 6)
+
+
+class AggregateAnnotationPruningTests(TestCase):
+    def test_unused_aliased_aggregate_pruned(self):
+        with CaptureQueriesContext(connection) as ctx:
+            Book.objects.alias(
+                authors_count=Count("authors"),
+            ).count()
+        sql = ctx.captured_queries[0]["sql"].lower()
+        self.assertEqual(sql.count("select"), 1, "No subquery wrapping required")
+        self.assertNotIn("authors_count", sql)
+
+    def test_non_aggregate_annotation_pruned(self):
+        with CaptureQueriesContext(connection) as ctx:
+            Book.objects.annotate(
+                name_lower=Lower("name"),
+            ).count()
+        sql = ctx.captured_queries[0]["sql"].lower()
+        self.assertEqual(sql.count("select"), 1, "No subquery wrapping required")
+        self.assertNotIn("name_lower", sql)
+
+    def test_unreferenced_aggregate_annotation_pruned(self):
+        with CaptureQueriesContext(connection) as ctx:
+            Book.objects.annotate(
+                authors_count=Count("authors"),
+            ).count()
+        sql = ctx.captured_queries[0]["sql"].lower()
+        self.assertEqual(sql.count("select"), 2, "Subquery wrapping required")
+        self.assertNotIn("authors_count", sql)
+
+    def test_referenced_aggregate_annotation_kept(self):
+        with CaptureQueriesContext(connection) as ctx:
+            Book.objects.annotate(
+                authors_count=Count("authors"),
+            ).aggregate(Avg("authors_count"))
+        sql = ctx.captured_queries[0]["sql"].lower()
+        self.assertEqual(sql.count("select"), 2, "Subquery wrapping required")
+        self.assertEqual(sql.count("authors_count"), 2)
`;

/**
 * SWE-bench Verified django__django-16263. Agent-visible material is frozen to
 * the issue discussion and the base revision. The merged source patch and the
 * held-out regression tests remain grader-only.
 */
export const djangoAnnotationPruningTask: RealOssTask = {
  instanceId: "django__django-16263",
  taskId: "task:real-oss:django__django-16263",
  codebaseId: "codebase:github:django/django@321ecb40",
  repository: "django/django",
  repositoryUrl: "https://github.com/django/django.git",
  license: "BSD-3-Clause",
  baseCommit: "321ecb40f4da842926e1bc07e11df4aabe53ca4b",
  upstreamIssueUrl: "https://code.djangoproject.com/ticket/28477",
  upstreamPullRequestUrl: "https://github.com/django/django/pull/16263",
  publicPrompt: `Strip unused annotations from count queries.

The query below produces a SQL statement that includes Count("chapters"),
despite not being used in any filter operations:

Book.objects.annotate(Count("chapters")).count()

It produces the same results as Book.objects.count(). Django should strip
annotations that are not referenced by filters, other annotations, or ordering.
Preserve annotations that are required by query semantics.`,
  acceptanceStatement:
    "Count and terminal aggregation queries omit unused annotations while retaining every annotation dependency required for correct results.",
  nonGoals: [
    "Editing tests or project metadata",
    "Changing public QuerySet APIs",
    "Removing annotations that are referenced by aggregates, predicates, or ordering",
  ],
  risk: "high",
  codeRoots: ["django"],
  allowedPaths: [
    "django/db/models/expressions.py",
    "django/db/models/query_utils.py",
    "django/db/models/sql/query.py",
    "django/db/models/sql/where.py",
  ],
  targets: [
    {
      workItemId: "work-item:django-expression-reference-discovery",
      plannedSymbolId: "planned-symbol:django:BaseExpression.get_refs",
      relativePath: "django/db/models/expressions.py",
      qualifiedName: "BaseExpression.get_refs",
      symbolKind: "method",
      binding: "planned",
      responsibility: "Expose annotation aliases referenced transitively by an expression tree.",
      ontologyNodeIds: [annotationDependencyNode],
      capabilityId: "capability:django-annotation-dependency-analysis",
    },
    {
      workItemId: "work-item:django-ref-reference-discovery",
      plannedSymbolId: "planned-symbol:django:Ref.get_refs",
      relativePath: "django/db/models/expressions.py",
      qualifiedName: "Ref.get_refs",
      symbolKind: "method",
      binding: "planned",
      responsibility: "Report the annotation alias represented by a Ref expression.",
      ontologyNodeIds: [annotationDependencyNode],
      dependsOn: ["work-item:django-expression-reference-discovery"],
      capabilityId: "capability:django-annotation-dependency-analysis",
    },
    {
      workItemId: "work-item:django-q-summary-resolution",
      plannedSymbolId: "planned-symbol:django:Q.resolve_expression",
      relativePath: "django/db/models/query_utils.py",
      qualifiedName: "Q.resolve_expression",
      symbolKind: "method",
      binding: "required",
      responsibility: "Preserve summary-mode annotation references while resolving Q predicates.",
      ontologyNodeIds: [annotationDependencyNode, queryPredicateNode],
      dependsOn: ["work-item:django-ref-reference-discovery"],
      capabilityId: "capability:django-summary-predicate-resolution",
    },
    {
      workItemId: "work-item:django-annotation-lookup-reference",
      plannedSymbolId: "planned-symbol:django:refs_expression",
      relativePath: "django/db/models/query_utils.py",
      qualifiedName: "refs_expression",
      symbolKind: "function",
      binding: "required",
      responsibility:
        "Identify the annotation alias matched by a lookup path without losing its identity.",
      ontologyNodeIds: [annotationDependencyNode, queryPredicateNode],
      capabilityId: "capability:django-summary-predicate-resolution",
    },
    {
      workItemId: "work-item:django-aggregation-query-planning",
      plannedSymbolId: "planned-symbol:django:Query.get_aggregation",
      relativePath: "django/db/models/sql/query.py",
      qualifiedName: "Query.get_aggregation",
      symbolKind: "method",
      binding: "required",
      responsibility:
        "Choose direct or subquery aggregation and retain only transitively required annotations.",
      ontologyNodeIds: [annotationDependencyNode, aggregationPlanningNode, queryPredicateNode],
      dependsOn: [
        "work-item:django-expression-reference-discovery",
        "work-item:django-ref-reference-discovery",
        "work-item:django-q-summary-resolution",
      ],
      capabilityId: "capability:django-aggregation-planning",
    },
    {
      workItemId: "work-item:django-summary-lookup-resolution",
      plannedSymbolId: "planned-symbol:django:Query.solve_lookup_type",
      relativePath: "django/db/models/sql/query.py",
      qualifiedName: "Query.solve_lookup_type",
      symbolKind: "method",
      binding: "required",
      responsibility:
        "Resolve annotation lookups as references when building a terminal summary query.",
      ontologyNodeIds: [annotationDependencyNode, queryPredicateNode],
      dependsOn: ["work-item:django-annotation-lookup-reference"],
      capabilityId: "capability:django-summary-predicate-resolution",
    },
    {
      workItemId: "work-item:django-summary-filter-building",
      plannedSymbolId: "planned-symbol:django:Query.build_filter",
      relativePath: "django/db/models/sql/query.py",
      qualifiedName: "Query.build_filter",
      symbolKind: "method",
      binding: "required",
      responsibility: "Propagate terminal-summary semantics through predicate construction.",
      ontologyNodeIds: [annotationDependencyNode, queryPredicateNode],
      dependsOn: ["work-item:django-summary-lookup-resolution"],
      capabilityId: "capability:django-summary-predicate-resolution",
    },
    {
      workItemId: "work-item:django-summary-q-building",
      plannedSymbolId: "planned-symbol:django:Query._add_q",
      relativePath: "django/db/models/sql/query.py",
      qualifiedName: "Query._add_q",
      symbolKind: "method",
      binding: "required",
      responsibility: "Propagate terminal-summary semantics across nested Q predicate trees.",
      ontologyNodeIds: [annotationDependencyNode, queryPredicateNode],
      dependsOn: ["work-item:django-summary-filter-building"],
      capabilityId: "capability:django-summary-predicate-resolution",
    },
    {
      workItemId: "work-item:django-where-reference-discovery",
      plannedSymbolId: "planned-symbol:django:WhereNode.get_refs",
      relativePath: "django/db/models/sql/where.py",
      qualifiedName: "WhereNode.get_refs",
      symbolKind: "method",
      binding: "planned",
      responsibility: "Expose annotation aliases referenced transitively by a predicate tree.",
      ontologyNodeIds: [annotationDependencyNode, queryPredicateNode],
      dependsOn: ["work-item:django-expression-reference-discovery"],
      capabilityId: "capability:django-annotation-dependency-analysis",
    },
  ],
  sourceIntegrity: [
    {
      relativePath: "LICENSE",
      sha256: "b846415d1b514e9c1dff14a22deb906d794bc546ca6129f950a18cd091e2a669",
    },
    {
      relativePath: "django/db/models/expressions.py",
      sha256: "d587eaab02c1a25d8b57096e9ac85f74b52de6a9e1375710b30ade4ca92bcadc",
    },
    {
      relativePath: "django/db/models/query_utils.py",
      sha256: "dde38fb8c9355dbc58ed4241fd03a83afcf3524b958a251335fd320dfdfdddd5",
    },
    {
      relativePath: "django/db/models/sql/query.py",
      sha256: "e8e8149c1994e52f53bd4e774c3d4ec6e8d79d1cba3661a2d594629e5b474fc2",
    },
    {
      relativePath: "django/db/models/sql/where.py",
      sha256: "4734ee3420579a6a355d66e7054f685348d1d5c6defee442e8924c2433f89f67",
    },
    {
      relativePath: "tests/aggregation/tests.py",
      sha256: "c94a03d79aede98d88b849f681d36cc33a511c0975d8ea6c28997c032d208cf1",
    },
  ],
  environment: {
    pythonVersion: "3.11",
    packages: ["asgiref==3.6.0", "sqlparse==0.4.3"],
  },
  publicTest: {
    command: "./.venv/bin/python",
    args: [
      "tests/runtests.py",
      "--verbosity",
      "1",
      "--parallel",
      "1",
      "aggregation",
      "aggregation_regress",
      "annotations",
      "queries",
      "lookup",
    ],
  },
  hiddenTest: {
    patch: hiddenTestPatch,
    patchSha256: "f988938ad1feeae064f11fa1498b37e1743750d362c41eb207931719d4a3e914",
    runner: {
      kind: "django-selectors",
      command: "./.venv/bin/python",
      args: ["tests/runtests.py", "--verbosity", "1", "--parallel", "1"],
    },
    failToPass: [
      "aggregation.tests.AggregateAnnotationPruningTests.test_non_aggregate_annotation_pruned",
      "aggregation.tests.AggregateAnnotationPruningTests.test_unreferenced_aggregate_annotation_pruned",
      "aggregation.tests.AggregateAnnotationPruningTests.test_unused_aliased_aggregate_pruned",
    ],
    passToPass: [
      "aggregation.tests.AggregateAnnotationPruningTests.test_referenced_aggregate_annotation_kept",
      "aggregation.tests.AggregateTestCase.test_add_implementation",
      "aggregation.tests.AggregateTestCase.test_aggregate_alias",
      "aggregation.tests.AggregateTestCase.test_aggregate_annotation",
      "aggregation.tests.AggregateTestCase.test_aggregate_in_order_by",
      "aggregation.tests.AggregateTestCase.test_aggregate_join_transform",
      "aggregation.tests.AggregateTestCase.test_aggregate_multi_join",
      "aggregation.tests.AggregateTestCase.test_aggregate_over_aggregate",
      "aggregation.tests.AggregateTestCase.test_aggregate_over_complex_annotation",
      "aggregation.tests.AggregateTestCase.test_aggregate_transform",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_after_annotation",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_compound_expression",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_expression",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_group_by",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_integer",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_not_in_aggregate",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_passed_another_aggregate",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_unset",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_unsupported_by_count",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_using_date_from_database",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_using_date_from_python",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_using_datetime_from_database",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_using_datetime_from_python",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_using_decimal_from_database",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_using_decimal_from_python",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_using_duration_from_database",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_using_duration_from_python",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_using_time_from_database",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_using_time_from_python",
      "aggregation.tests.AggregateTestCase.test_aggregation_default_zero",
      "aggregation.tests.AggregateTestCase.test_aggregation_exists_annotation",
      "aggregation.tests.AggregateTestCase.test_aggregation_exists_multivalued_outeref",
      "aggregation.tests.AggregateTestCase.test_aggregation_expressions",
      "aggregation.tests.AggregateTestCase.test_aggregation_filter_exists",
      "aggregation.tests.AggregateTestCase.test_aggregation_nested_subquery_outerref",
      "aggregation.tests.AggregateTestCase.test_aggregation_order_by_not_selected_annotation_values",
      "aggregation.tests.AggregateTestCase.test_aggregation_random_ordering",
      "aggregation.tests.AggregateTestCase.test_aggregation_subquery_annotation",
      "aggregation.tests.AggregateTestCase.test_aggregation_subquery_annotation_exists",
      "aggregation.tests.AggregateTestCase.test_aggregation_subquery_annotation_multivalued",
      "aggregation.tests.AggregateTestCase.test_aggregation_subquery_annotation_related_field",
      "aggregation.tests.AggregateTestCase.test_aggregation_subquery_annotation_values",
      "aggregation.tests.AggregateTestCase.test_aggregation_subquery_annotation_values_collision",
      "aggregation.tests.AggregateTestCase.test_alias_sql_injection",
      "aggregation.tests.AggregateTestCase.test_annotate_basic",
      "aggregation.tests.AggregateTestCase.test_annotate_defer",
      "aggregation.tests.AggregateTestCase.test_annotate_defer_select_related",
      "aggregation.tests.AggregateTestCase.test_annotate_m2m",
      "aggregation.tests.AggregateTestCase.test_annotate_ordering",
      "aggregation.tests.AggregateTestCase.test_annotate_over_annotate",
      "aggregation.tests.AggregateTestCase.test_annotate_values",
      "aggregation.tests.AggregateTestCase.test_annotate_values_aggregate",
      "aggregation.tests.AggregateTestCase.test_annotate_values_list",
      "aggregation.tests.AggregateTestCase.test_annotated_aggregate_over_annotated_aggregate",
      "aggregation.tests.AggregateTestCase.test_annotation",
      "aggregation.tests.AggregateTestCase.test_annotation_expressions",
      "aggregation.tests.AggregateTestCase.test_arguments_must_be_expressions",
      "aggregation.tests.AggregateTestCase.test_avg_decimal_field",
      "aggregation.tests.AggregateTestCase.test_avg_duration_field",
      "aggregation.tests.AggregateTestCase.test_backwards_m2m_annotate",
      "aggregation.tests.AggregateTestCase.test_coalesced_empty_result_set",
      "aggregation.tests.AggregateTestCase.test_combine_different_types",
      "aggregation.tests.AggregateTestCase.test_complex_aggregations_require_kwarg",
      "aggregation.tests.AggregateTestCase.test_complex_values_aggregation",
      "aggregation.tests.AggregateTestCase.test_count",
      "aggregation.tests.AggregateTestCase.test_count_distinct_expression",
      "aggregation.tests.AggregateTestCase.test_count_star",
      "aggregation.tests.AggregateTestCase.test_dates_with_aggregation",
      "aggregation.tests.AggregateTestCase.test_decimal_max_digits_has_no_effect",
      "aggregation.tests.AggregateTestCase.test_distinct_on_aggregate",
      "aggregation.tests.AggregateTestCase.test_empty_aggregate",
      "aggregation.tests.AggregateTestCase.test_empty_result_optimization",
      "aggregation.tests.AggregateTestCase.test_even_more_aggregate",
      "aggregation.tests.AggregateTestCase.test_exists_extra_where_with_aggregate",
      "aggregation.tests.AggregateTestCase.test_exists_none_with_aggregate",
      "aggregation.tests.AggregateTestCase.test_expression_on_aggregation",
      "aggregation.tests.AggregateTestCase.test_filter_aggregate",
      "aggregation.tests.AggregateTestCase.test_filter_in_subquery_or_aggregation",
      "aggregation.tests.AggregateTestCase.test_filtering",
      "aggregation.tests.AggregateTestCase.test_fkey_aggregate",
      "aggregation.tests.AggregateTestCase.test_group_by_exists_annotation",
      "aggregation.tests.AggregateTestCase.test_group_by_subquery_annotation",
      "aggregation.tests.AggregateTestCase.test_grouped_annotation_in_group_by",
      "aggregation.tests.AggregateTestCase.test_more_aggregation",
      "aggregation.tests.AggregateTestCase.test_multi_arg_aggregate",
      "aggregation.tests.AggregateTestCase.test_multiple_aggregates",
      "aggregation.tests.AggregateTestCase.test_non_grouped_annotation_not_in_group_by",
      "aggregation.tests.AggregateTestCase.test_nonaggregate_aggregation_throws",
      "aggregation.tests.AggregateTestCase.test_nonfield_annotation",
      "aggregation.tests.AggregateTestCase.test_order_of_precedence",
      "aggregation.tests.AggregateTestCase.test_related_aggregate",
      "aggregation.tests.AggregateTestCase.test_reverse_fkey_annotate",
      "aggregation.tests.AggregateTestCase.test_single_aggregate",
      "aggregation.tests.AggregateTestCase.test_sum_distinct_aggregate",
      "aggregation.tests.AggregateTestCase.test_sum_duration_field",
      "aggregation.tests.AggregateTestCase.test_ticket11881",
      "aggregation.tests.AggregateTestCase.test_ticket12886",
      "aggregation.tests.AggregateTestCase.test_ticket17424",
      "aggregation.tests.AggregateTestCase.test_values_aggregation",
      "aggregation.tests.AggregateTestCase.test_values_annotation_with_expression",
    ],
  },
  sourceDocuments: [
    {
      documentId: "django-ticket:28477:description",
      kind: "issue",
      title: "Strip unused annotations from count queries",
      body: `Book.objects.annotate(Count("chapters")).count() includes the
Count("chapters") annotation even though no filter uses it and returns the same
result as Book.objects.count(). Annotations not referenced by filters, other
annotations, or ordering should be stripped from count queries. select_related
already has precedent for being ignored by count().`,
      sourceUrl: "https://code.djangoproject.com/ticket/28477",
      sourceSpan: "ticket #28477 description",
      observedAt: "2022-11-03T18:57:33.000Z",
      ontologyNodeIds: [annotationDependencyNode, aggregationPlanningNode],
    },
    {
      documentId: "django-ticket:28477:history-before-fix",
      kind: "issue",
      title: "Ticket #28477 pre-fix discussion",
      body: `The same pruning opportunity may apply to exists(). A reported
non-aggregate annotation made count() generate a subquery and GROUP BY, which
was much slower on large result sets. Calling values("pk") first was a partial
workaround. An earlier change removed the GROUP BY for one case but still left
the unused annotation in a subquery. Related discussion notes that subquery
annotations cannot always be excluded from GROUP BY safely.`,
      sourceUrl: "https://code.djangoproject.com/ticket/28477",
      sourceSpan: "ticket #28477 comments available before PR #16263",
      observedAt: "2022-11-03T18:57:33.000Z",
      ontologyNodeIds: [annotationDependencyNode, aggregationPlanningNode],
    },
    {
      documentId: "django-ticket:28477:accepted-design",
      kind: "issue",
      title: "Ticket #28477 accepted aggregation-pruning design",
      body: `The revised implementation applies to every use of aggregate(),
not only count(). Instead of systematically wrapping whenever existing
annotations are present, it wraps only when pre-existing selected annotations
are aggregate or window expressions, or another existing query constraint
requires a subquery. In both the wrapping and direct-aggregation paths,
annotations not referenced by the requested terminal aggregates are stripped.`,
      sourceUrl: "https://code.djangoproject.com/ticket/28477#comment:13",
      sourceSpan: "ticket #28477 comment 13 accepted design before merge",
      observedAt: "2022-11-06T08:55:46.630Z",
      ontologyNodeIds: [annotationDependencyNode, aggregationPlanningNode, queryPredicateNode],
    },
    {
      documentId: "django-docs:queryset-annotations-aliases:321ecb40",
      kind: "documentation",
      title: "QuerySet annotate() and alias() semantics",
      body: `annotate() adds an expression-derived value to each returned
QuerySet object and is not terminal; later filter(), order_by(), or annotate()
calls may depend on it. alias() saves an expression for later reuse without
selecting its value. Not selecting an unused aliased value removes redundant
database work, but an alias used by filtering, ordering, or a complex
expression remains semantically significant. An alias must be promoted to an
annotation before aggregate() can consume it. Selected pre-existing annotations
are therefore distinct from non-selected aliases: an unused alias alone does
not request a selected value.`,
      sourceUrl:
        "https://github.com/django/django/blob/321ecb40f4da842926e1bc07e11df4aabe53ca4b/docs/ref/models/querysets.txt#L260-L335",
      sourceSpan: "docs/ref/models/querysets.txt:260-335 at base commit",
      observedAt: "2022-11-03T18:57:33.000Z",
      ontologyNodeIds: [annotationDependencyNode, queryPredicateNode],
    },
    {
      documentId: "django-docs:aggregation:321ecb40",
      kind: "documentation",
      title: "Django aggregation topic guide",
      body: `annotate() computes a per-object summary and returns a QuerySet;
aggregate() computes a terminal summary. Because annotations can feed filters,
ordering, later annotations, and terminal aggregates, their dependency order
is significant. Combining aggregations and joins can also change results, so
query-shape optimizations must preserve aggregate semantics.`,
      sourceUrl:
        "https://github.com/django/django/blob/321ecb40f4da842926e1bc07e11df4aabe53ca4b/docs/topics/db/aggregation.txt#L150-L205",
      sourceSpan: "docs/topics/db/aggregation.txt:150-205 at base commit",
      observedAt: "2022-11-03T18:57:33.000Z",
      ontologyNodeIds: [annotationDependencyNode, aggregationPlanningNode],
    },
    {
      documentId: "django-source:query-get-aggregation:321ecb40",
      kind: "documentation",
      title: "Base-revision aggregation planner comments",
      body: `get_aggregation() must produce one terminal result. The base
implementation uses a subquery when GROUP BY, slicing, existing annotations,
distinct, or set operations require inner-query semantics. Limit and distinct
must be applied inside the subquery. When the inner query otherwise selects no
field, it selects the model primary key so the subquery remains valid.`,
      sourceUrl:
        "https://github.com/django/django/blob/321ecb40f4da842926e1bc07e11df4aabe53ca4b/django/db/models/sql/query.py#L438-L537",
      sourceSpan: "django/db/models/sql/query.py:438-537 comments at base commit",
      observedAt: "2022-11-03T18:57:33.000Z",
      ontologyNodeIds: [aggregationPlanningNode],
    },
  ],
  normativeRecords: [
    {
      kind: "decision",
      recordId: "decision:django-prune-unused-terminal-annotations",
      revisionId: "decision:django-prune-unused-terminal-annotations@ticket-28477",
      statement:
        "Terminal count and aggregation planning removes annotations that are not transitively referenced by the requested summary, predicates, or ordering.",
      evidenceIds: [
        "django-ticket:28477:description",
        "django-ticket:28477:history-before-fix",
        "django-ticket:28477:accepted-design",
        "django-docs:queryset-annotations-aliases:321ecb40",
      ],
      ontologyNodeIds: [annotationDependencyNode, aggregationPlanningNode, queryPredicateNode],
    },
    {
      kind: "decision",
      recordId: "decision:django-preserve-required-aggregation-subqueries",
      revisionId: "decision:django-preserve-required-aggregation-subqueries@321ecb40",
      statement:
        "Aggregation uses a subquery for grouping, slicing, distinctness, set operations, selected pre-existing aggregate or window annotations, or aggregate/window predicates; an ordinary annotation or non-selected alias alone does not trigger wrapping.",
      evidenceIds: [
        "django-ticket:28477:accepted-design",
        "django-docs:aggregation:321ecb40",
        "django-source:query-get-aggregation:321ecb40",
      ],
      ontologyNodeIds: [aggregationPlanningNode, queryPredicateNode],
    },
    {
      kind: "domain_term",
      recordId: "domain-term:django-annotation",
      revisionId: "domain-term:django-annotation@321ecb40",
      term: "Annotation",
      definition:
        "A named query expression whose value is added per QuerySet object and may be consumed by later predicates, ordering, annotations, or terminal aggregation.",
      avoid: ["always-selected column", "terminal aggregate result"],
      evidenceIds: [
        "django-docs:queryset-annotations-aliases:321ecb40",
        "django-docs:aggregation:321ecb40",
      ],
      ontologyNodeIds: [annotationDependencyNode, aggregationPlanningNode],
    },
    {
      kind: "domain_term",
      recordId: "domain-term:django-alias",
      revisionId: "domain-term:django-alias@321ecb40",
      term: "Alias",
      definition:
        "A named stored expression available for later query operations whose value is not selected unless it is promoted to an annotation.",
      avoid: ["unused annotation", "selected result column"],
      evidenceIds: ["django-docs:queryset-annotations-aliases:321ecb40"],
      ontologyNodeIds: [annotationDependencyNode, queryPredicateNode],
    },
    {
      kind: "domain_term",
      recordId: "domain-term:django-summary-aggregate",
      revisionId: "domain-term:django-summary-aggregate@321ecb40",
      term: "Summary aggregate",
      definition:
        "A terminal aggregate expression that returns one summary value and may depend on a previously named annotation.",
      avoid: ["per-object annotation"],
      evidenceIds: [
        "django-docs:aggregation:321ecb40",
        "django-source:query-get-aggregation:321ecb40",
      ],
      ontologyNodeIds: [annotationDependencyNode, aggregationPlanningNode],
    },
    {
      kind: "invariant",
      recordId: "invariant:django-transitive-annotation-dependencies-survive",
      revisionId: "invariant:django-transitive-annotation-dependencies-survive@ticket-28477",
      statement:
        "If a retained aggregate, annotation, predicate, or ordering expression references an annotation alias, that alias and its transitive annotation dependencies remain available to query compilation.",
      evidenceIds: [
        "django-ticket:28477:description",
        "django-docs:queryset-annotations-aliases:321ecb40",
      ],
      ontologyNodeIds: [annotationDependencyNode, aggregationPlanningNode, queryPredicateNode],
    },
    {
      kind: "invariant",
      recordId: "invariant:django-predicate-annotation-dependencies-survive",
      revisionId: "invariant:django-predicate-annotation-dependencies-survive@ticket-28477",
      statement:
        "Annotations referenced through WHERE, HAVING, QUALIFY, or nested Q predicates are not pruned from a terminal aggregation query.",
      evidenceIds: [
        "django-ticket:28477:description",
        "django-docs:queryset-annotations-aliases:321ecb40",
      ],
      ontologyNodeIds: [annotationDependencyNode, queryPredicateNode],
    },
    {
      kind: "invariant",
      recordId: "invariant:django-subquery-trigger-is-semantic",
      revisionId: "invariant:django-subquery-trigger-is-semantic@ticket-28477-comment-13",
      statement:
        "A selected unused aggregate annotation still requires aggregation subquery semantics but is omitted from the inner SELECT; a non-aggregate annotation and a non-selected aggregate alias do not by themselves require a subquery.",
      evidenceIds: [
        "django-ticket:28477:description",
        "django-ticket:28477:accepted-design",
        "django-docs:queryset-annotations-aliases:321ecb40",
      ],
      ontologyNodeIds: [annotationDependencyNode, aggregationPlanningNode],
    },
    {
      kind: "invariant",
      recordId: "invariant:django-subquery-mask-keeps-summary-dependencies",
      revisionId:
        "invariant:django-subquery-mask-keeps-summary-dependencies@ticket-28477-comment-13",
      statement:
        "When aggregation uses an inner subquery, its annotation mask contains the requested terminal aggregate names and the annotation aliases transitively referenced by those aggregates, not every pre-existing selected annotation.",
      evidenceIds: [
        "django-ticket:28477:accepted-design",
        "django-source:query-get-aggregation:321ecb40",
      ],
      ontologyNodeIds: [annotationDependencyNode, aggregationPlanningNode],
    },
    {
      kind: "invariant",
      recordId: "invariant:django-direct-aggregation-inlines-annotation-references",
      revisionId:
        "invariant:django-direct-aggregation-inlines-annotation-references@ticket-28477-comment-13",
      statement:
        "When aggregation stays on the direct query, references from requested terminal aggregates to existing annotations are replaced by their underlying expressions before existing annotations are masked out.",
      evidenceIds: [
        "django-ticket:28477:accepted-design",
        "django-docs:queryset-annotations-aliases:321ecb40",
        "django-source:query-get-aggregation:321ecb40",
      ],
      ontologyNodeIds: [annotationDependencyNode, aggregationPlanningNode],
    },
    {
      kind: "invariant",
      recordId: "invariant:django-query-semantics-before-query-shape",
      revisionId: "invariant:django-query-semantics-before-query-shape@321ecb40",
      statement:
        "Removing an annotation or subquery is valid only when count and aggregate results, slicing, distinctness, grouping, and predicate behavior remain unchanged.",
      evidenceIds: [
        "django-ticket:28477:history-before-fix",
        "django-docs:aggregation:321ecb40",
        "django-source:query-get-aggregation:321ecb40",
      ],
      ontologyNodeIds: [aggregationPlanningNode, queryPredicateNode],
    },
  ],
  limitations: [
    "The Kontext arm receives nine ontology-linked behavior-bearing targets derived from the merged change shape; this is intentional planning guidance and gives it more structure than raw RAG, while no source patch is exposed.",
    "The frozen public corpus is a curated, timestamp-bounded projection of the issue, base documentation, and base code comments rather than a live repository search.",
  ],
};
