// Make each Graphviz diagram a link that opens the diagram on its own, full size,
// in a new tab, where the browser can zoom it.  The diagram is an <object>, which
// takes clicks itself, so the clicks are turned off and the link around it gets them.
document.addEventListener("DOMContentLoaded", function () {
  document.querySelectorAll("div.graphviz > object[type='image/svg+xml']").forEach(function (obj) {
    var link = document.createElement("a");
    link.href = obj.data;
    link.target = "_blank";
    link.rel = "noopener";
    link.title = "Click to open the diagram full size in a new tab";
    link.style.display = "block";
    link.style.cursor = "zoom-in";
    obj.parentNode.insertBefore(link, obj);
    link.appendChild(obj);
    obj.style.pointerEvents = "none";
    obj.style.maxWidth = "100%";
  });
});
