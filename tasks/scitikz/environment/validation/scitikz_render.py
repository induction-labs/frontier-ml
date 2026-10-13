"""Paper-compatible pdfLaTeX rendering; official grading requires the isolated launcher."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import subprocess
import tempfile

MAX_ANSWER = 64000


def prepare_devices(root):
    """Device nodes are allowed in Modal sandboxes, but not in image-build containers."""
    for name, minor in (("null", 3), ("urandom", 9), ("random", 8)):
        path = Path(root)/"dev"/name
        try:
            os.mknod(path, stat.S_IFCHR | 0o666, os.makedev(1, minor))
        except FileExistsError:
            pass
        info = path.lstat()
        if not stat.S_ISCHR(info.st_mode) or info.st_rdev != os.makedev(1, minor):
            raise RuntimeError("Invalid device in isolated TeX toolchain")
        os.chmod(path, 0o666)


def extract_tex(answer):
    if not isinstance(answer, str) or len(answer) > MAX_ANSWER:
        return None, "response_size"
    match = re.search(r"```latex\s*(.*?)```", answer, re.I | re.S)
    if not match or not match.group(1).strip():
        return None, "missing_closed_latex_fence"
    return match.group(1).strip(), None


def code_for_metrics(answer):
    """The author's renderer saves truncated open blocks for code-only metrics."""
    if len(answer) > MAX_ANSWER:
        return ""
    match = re.search(r"```latex\s*(.*?)(?:```|\Z)", answer, re.I | re.S)
    return match.group(1).strip() if match else ""


def render_tex(source, destination, index, config):
    from PIL import Image
    jail = config.get("tex_root")
    launcher = config.get("tex_launcher")
    if config.get("require_isolation") and (not jail or not launcher or os.geteuid() != 0):
        raise RuntimeError("Official TeX rendering requires the root-owned isolated launcher")
    if jail:
        prepare_devices(jail)
    parent = Path(jail)/"jobs" if jail else None
    with tempfile.TemporaryDirectory(prefix="tikz-", dir=parent) as temporary:
        work = Path(temporary)
        uid = 20000 + index
        if jail:
            os.chown(work, uid, uid)
        match = re.search(r"\\documentclass(?:\[[^\]]*\])?\{([^}]+)\}", source)
        standalone = bool(match and "standalone" in match.group(1).lower())
        rendered_source = source
        if not standalone and r"\pagestyle{empty}" not in source and r"\thispagestyle{empty}" not in source:
            rendered_source = source.replace(r"\begin{document}", r"\begin{document}"+"\n"+r"\pagestyle{empty}\thispagestyle{empty}", 1)
        (work/"figure.tex").write_text(rendered_source)
        if jail:
            os.chown(work/"figure.tex", uid, uid)

        def command(args, seconds):
            executable = shutil.which(args[0]) if not jail else "/usr/bin/"+args[0]
            if not executable:
                raise RuntimeError(f"Required renderer binary is missing: {args[0]}")
            argv = [executable, *args[1:]]
            if jail:
                argv = [launcher, str(jail), str(uid), "/jobs/"+work.name, *argv]
            env = {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": str(work), "TMPDIR": str(work),
                   "LANG": "C.UTF-8", "openin_any": "p", "openout_any": "p",
                   "OMP_NUM_THREADS": "1", "MAGICK_THREAD_LIMIT": "1"}
            with (work/"command.log").open("wb") as log:
                proc = subprocess.Popen(argv, cwd=work, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                try:
                    code = proc.wait(timeout=seconds)
                except subprocess.TimeoutExpired:
                    code = -1
                finally:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
            if code == 125:
                raise RuntimeError("TeX isolation launcher failed: "+(work/"command.log").read_text(errors="replace")[-2000:])
            return code == 0

        if not command(["pdflatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "figure.tex"], 30):
            return {"compiled": False, "failure": "latex_compile", "log": (work/"command.log").read_text(errors="replace")[-1500:]}
        if not (work/"figure.pdf").is_file():
            return {"compiled": False, "failure": "missing_pdf"}
        pdf, fallback = "figure.pdf", False
        if not standalone:
            if command(["pdfcrop", "--margins", "10", pdf, "cropped.pdf"], 300):
                pdf = "cropped.pdf"
            else:
                fallback = True
        crop_options = ["-fuzz", "1%", "-trim", "+repage", "-bordercolor", "white", "-border", "42"] if fallback else []
        if not command(["convert", "-limit", "memory", "256MiB", "-limit", "map", "512MiB", "-limit", "disk", "512MiB",
                        "-density", "300", pdf+"[0]", "-background", "white", "-flatten", *crop_options,
                        "-quality", "95", "figure.png"], 60):
            return {"compiled": False, "failure": "rasterize", "log": (work/"command.log").read_text(errors="replace")[-1500:]}
        try:
            image = Image.open(work/"figure.png")
            if max(image.size) > 8192 or image.width*image.height > 32_000_000:
                return {"compiled": False, "failure": "raster_size"}
            image.convert("RGB").save(destination)
        except (OSError, Image.DecompressionBombError):
            return {"compiled": False, "failure": "invalid_raster"}
        return {"compiled": True}


def render_answers(predictions, directory, config):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    def one(item):
        index, prediction = item
        source, failure = extract_tex(prediction["answer"])
        if failure:
            return {"id": prediction["id"], "compiled": False, "failure": failure,
                    "source": code_for_metrics(prediction["answer"])}
        png = directory/f"{index:04d}.png"
        result = render_tex(source, png, index, config)
        result.update(id=prediction["id"], source=source)
        if result["compiled"]:
            result["png"] = str(png)
        return result
    with ThreadPoolExecutor(max_workers=config.get("render_workers", 4)) as pool:
        return list(pool.map(one, enumerate(predictions)))
