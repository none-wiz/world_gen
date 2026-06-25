import marimo

__generated_with = "0.23.9"
app = marimo.App(
    width="medium",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell
def _():
    import marimo as mo
    import os
    import shutil
    import io

    # Create the text input
    dir_input = mo.ui.text(
        value="", 
        placeholder="e.g., my_folder", 
        label="Enter directory name (relative to current path):"
    )

    mo.vstack([
        mo.md("### 📁 Download a Whole Directory"),
        dir_input
    ])
    return dir_input, io, mo, os


@app.cell
def _(dir_input, io, mo, os):
    # Marimo will safely react to dir_input.value here because it's a separate cell
    target_dir = dir_input.value

    if not target_dir:
        output = mo.md("💡 *Enter a directory name above to get started.*")
    elif not os.path.exists(target_dir):
        output = mo.md(f"❌ **Error:** The directory `{target_dir}` does not exist.")
    elif not os.path.isdir(target_dir):
        output = mo.md(f"❌ **Error:** `{target_dir}` is a file, not a directory.")
    else:
        # Create the zip in memory
        zip_buffer = io.BytesIO()
        import zipfile
    
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(target_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    # Keep the folder structure inside the zip relative
                    arcname = os.path.relpath(file_path, os.path.dirname(target_dir))
                    zip_file.write(file_path, arcname)
                
        zip_buffer.seek(0)
    
        # FIX: Changed mo.ui.download to mo.download
        output = mo.download(
            data=zip_buffer,
            filename=f"{os.path.basename(os.path.normpath(target_dir))}.zip",
            label=f"📦 Download {target_dir} as ZIP",
        )

    output
    return


if __name__ == "__main__":
    app.run()
