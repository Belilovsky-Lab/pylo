import os,site,re

def patch_mup_layer():
    """Patch mup/layer.py to modify the forward method"""
    # Find all site-packages directories
    site_packages = site.getsitepackages()
    patched = False

    # Look for mup/layer.py in each site-packages directory
    for sp in site_packages:
        layer_path = os.path.join(sp, "mup", "layer.py")
        if os.path.exists(layer_path):
            print(f"Found mup/layer.py at {layer_path}")

            # Read the file content
            with open(layer_path, "r") as f:
                content = f.read()

            # Define the pattern to find and the replacement
            old_forward = r"def forward\(self, x\):\s+return super\(\)\.forward\(\s+self\.output_mult \* x / self\.width_mult\(\)\)"
            new_forward = """def forward(self, x):
        # return super().forward(
        #     self.output_mult * x / self.width_mult())
        return super().forward(x) * (self.output_mult / self.width_mult())"""

            # Replace the pattern
            modified_content = re.sub(
                old_forward, new_forward, content, flags=re.DOTALL
            )

            # Check if any changes were made
            if modified_content == content:
                print(
                    "Warning: Pattern not found in the file. The file might have a different format."
                )
                return False

            # Write the modified content back to the file
            with open(layer_path, "w") as f:
                f.write(modified_content)

            print(f"Successfully patched mup/layer.py")
            patched = True
            break

    if not patched:
        print("Could not find mup/layer.py in any site-packages directory.")
        return False

    return True


if __name__ == "__main__":
    success = patch_mup_layer()
    if success:
        print("Patch applied successfully.")
    else:
        print("Failed to apply patch.")