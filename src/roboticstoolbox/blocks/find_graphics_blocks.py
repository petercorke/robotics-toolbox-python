import ast
import glob
import os

path = "/Users/pic/Library/CloudStorage/Dropbox/code/robotics-toolbox-python/src/roboticstoolbox/blocks/*.py"

for filename in glob.glob(path):
    with open(filename, "r") as f:
        try:
            tree = ast.parse(f.read(), filename=filename)
        except Exception as e:
            print(f"Error parsing {filename}: {e}")
            continue

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            is_graphics_block = False
            for base in node.bases:
                if isinstance(base, ast.Name) and base.id == "GraphicsBlock":
                    is_graphics_block = True
                    break
                elif isinstance(base, ast.Attribute) and base.attr == "GraphicsBlock":
                    is_graphics_block = True
                    break
            
            if is_graphics_block:
                try:
                    with open(filename, "r") as f:
                        lines = f.readlines()
                    class_lines = lines[node.lineno-1 : node.end_lineno]
                    class_text = "".join(class_lines)
                    
                    has_start = "super().start(" in class_text
                    has_step = "super().step(" in class_text
                    has_done = "super().done(" in class_text
                    
                    print(f"{filename}:{node.lineno} Class {node.name}: start={has_start}, step={has_step}, done={has_done}")
                except Exception as e:
                     print(f"Error processing class {node.name} in {filename}: {e}")
