import ast
import builtins
import keyword
import tokenize
import io


class ScopedRenamer(ast.NodeTransformer):
    """Rename user-defined identifiers to generic names while respecting scopes."""

    def __init__(self):
        self.counter = 0

        # stack of scope dictionaries
        self.scopes = [{}]

        self.reserved = set(dir(builtins)) | set(keyword.kwlist)

    # ---------- utilities ----------

    def fresh_name(self):
        name = f"v{self.counter}"
        self.counter += 1
        return name

    def current_scope(self):
        return self.scopes[-1]

    def define(self, name):
        if name in self.reserved or name.startswith("__"):
            return name

        scope = self.current_scope()

        if name not in scope:
            scope[name] = self.fresh_name()

        return scope[name]

    def resolve(self, name):
        if name in self.reserved or name.startswith("__"):
            return name

        # search inner -> outer scopes
        for scope in reversed(self.scopes):
            if name in scope:
                return scope[name]

        return name

    def push_scope(self):
        self.scopes.append({})

    def pop_scope(self):
        self.scopes.pop()

    def visit_Import(self, node):
        return node

    def visit_ImportFrom(self, node):
        return node

    def visit_ClassDef(self, node):
        node.name = self.define(node.name)

        self.push_scope()

        self.generic_visit(node)

        self.pop_scope()

        return node

    def visit_FunctionDef(self, node):
        node.name = self.define(node.name)

        self.push_scope()

        # rename parameters
        for arg in node.args.args:
            arg.arg = self.define(arg.arg)

        for arg in node.args.posonlyargs:
            arg.arg = self.define(arg.arg)

        for arg in node.args.kwonlyargs:
            arg.arg = self.define(arg.arg)

        if node.args.vararg:
            node.args.vararg.arg = self.define(node.args.vararg.arg)

        if node.args.kwarg:
            node.args.kwarg.arg = self.define(node.args.kwarg.arg)

        self.generic_visit(node)

        self.pop_scope()

        return node

    def visit_AsyncFunctionDef(self, node):
        return self.visit_FunctionDef(node)

    def visit_Assign(self, node):
        for target in node.targets:
            self.handle_target(target)

        self.generic_visit(node)

        return node

    def visit_AnnAssign(self, node):
        self.handle_target(node.target)

        self.generic_visit(node)

        return node

    def visit_For(self, node):
        self.handle_target(node.target)

        self.generic_visit(node)

        return node

    def visit_comprehension(self, node):
        self.handle_target(node.target)

        self.generic_visit(node)

        return node

    def handle_target(self, target):
        if isinstance(target, ast.Name):
            target.id = self.define(target.id)

        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self.handle_target(elt)

    def visit_Name(self, node):
        if isinstance(node.ctx, ast.Load):
            node.id = self.resolve(node.id)

        return node

    def visit_Attribute(self, node):
        self.generic_visit(node)
        return node

    def visit_AugAssign(self, node):
        self.handle_target(node.target)
        self.generic_visit(node)
        return node

    def visit_With(self, node: ast.With):
        for item in node.items:
            if item.optional_vars:
                self.handle_target(item.optional_vars)

        self.generic_visit(node)
        return node

    def visit_AsyncWith(self, node: ast.AsyncWith):
        for item in node.items:
            if item.optional_vars:
                self.handle_target(item.optional_vars)

        self.generic_visit(node)
        return node

    def visit_ExceptHandler(self, node: ast.ExceptHandler):
        if node.name:
            node.name = self.define(node.name)

        self.generic_visit(node)
        return node


class DocstringRemover(ast.NodeTransformer):

    @staticmethod
    def is_docstring(node: ast.AST) -> bool:
        return (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        )

    def strip_docstring(self, node: ast.AST) -> ast.AST:
        if node.body and self.is_docstring(node.body[0]):
            node.body = node.body[1:]

        return node

    def visit_Module(self, node: ast.Module):
        node = self.strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        node = self.strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        node = self.strip_docstring(node)
        self.generic_visit(node)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        node = self.strip_docstring(node)
        self.generic_visit(node)
        return node


class CommentRemover:

    @staticmethod
    def remove_comments(code: str) -> str:
        result: list[tokenize.TokenInfo] = []
        tokens = tokenize.generate_tokens(io.StringIO(code).readline)
        for token in tokens:
            if token.type != tokenize.COMMENT:
                result.append(token)
        return tokenize.untokenize(result)


def normalize_identifiers(code: str) -> str:
    tree = ast.parse(code)

    renamer = ScopedRenamer()

    tree = renamer.visit(tree)

    ast.fix_missing_locations(tree)

    return ast.unparse(tree)


def remove_docstrings(code: str) -> str:
    tree = ast.parse(code)

    remover = DocstringRemover()

    tree = remover.visit(tree)
    ast.fix_missing_locations(tree)

    return ast.unparse(tree)


def remove_comments(code: str) -> str:
    return CommentRemover.remove_comments(code)


def example():
    code = '''
"""module docstring (should be removed)"""

import numpy as np

class Model:
    """class docstring"""

    def __init__(self, data):
        """init docstring"""
        self.data = data
        self.cache = {}

    def update(self, x):
        # update comment
        self.cache[x] = x * 2
        return self.cache[x]

    def compute(self, values):
        """compute docstring"""
        total = 0

        # loop with shadowed variable
        for x in values:
            total += x

        # comprehension (new scope)
        squares = [x * x for x in values]

        # lambda (new scope)
        f = lambda x: x + 1

        return np.mean(squares)


def outer(a, b):
    """outer docstring"""

    def inner(a):
        # nested function scope
        return a * 2

    try:
        result = inner(a)
    except Exception as e:
        result = b

    return result


# augmented assignment test
counter = 0
counter += 1

# tuple unpacking
x, y = 1, 2

# with statement
with open("file.txt") as f:
    content = f.read()
'''
    print("Original code:")
    print(code)

    print("\nNormalized identifiers:")
    print(normalize_identifiers(code))

    print("\nRemoved docstrings:")
    print(remove_docstrings(code))

    print("\nRemoved comments:")
    print(remove_comments(code))


if __name__ == "__main__":
    example()
