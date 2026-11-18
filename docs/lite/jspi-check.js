// Warn, before the Python kernel starts, if the browser cannot run it.
//
// The Pyodide kernel waits for asynchronous work (package installs, the plotting
// backend) using WebAssembly JSPI, JavaScript Promise Integration.  Without it the
// kernel does not start or never finishes a cell, and the page just shows a spinner.
// Chrome and Edge have had JSPI for a while, Firefox has it from 153 and Safari from 27.  This banner
// says so, instead of leaving the reader to guess.  It is added to each JupyterLite
// page by inject_jspi_check.py after the site is built.
(function () {
  "use strict";
  if (typeof WebAssembly !== "undefined" && typeof WebAssembly.Suspending === "function") {
    return;
  }
  function show() {
    var bar = document.createElement("div");
    bar.setAttribute("role", "alert");
    bar.style.cssText =
      "position:fixed;top:0;left:0;right:0;z-index:2147483647;padding:12px 16px;" +
      "background:#fff3cd;color:#3d2e00;border-bottom:2px solid #b58900;" +
      "font:15px/1.4 system-ui,sans-serif";
    bar.textContent =
      "This browser cannot run the Python notebooks here: it lacks WebAssembly " +
      "JSPI (JavaScript Promise Integration), so the kernel will hang or fail to start. " +
      "Use a current Chrome or Edge, Firefox 153 or later, or Safari 27 or later.";
    document.body.appendChild(bar);
  }
  if (document.body) {
    show();
  } else {
    document.addEventListener("DOMContentLoaded", show);
  }
})();
