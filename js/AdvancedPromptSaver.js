import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";

app.registerExtension({
    name: "AdvancedPromptSaver",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AdvancedPromptSaver") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);

                // Create widget
                const statusWidget = ComfyWidgets["STRING"](this, "status_display", ["STRING", { multiline: true }], app).widget;
                statusWidget.inputEl.readOnly = true;
                statusWidget.inputEl.style.opacity = 0.8;
                statusWidget.inputEl.style.backgroundColor = "#2a2a2a";
                statusWidget.inputEl.style.border = "1px solid #555";
                statusWidget.inputEl.style.borderRadius = "4px";
                statusWidget.inputEl.style.fontSize = "10px";
                statusWidget.inputEl.style.fontFamily = "monospace";
                statusWidget.inputEl.style.color = "#fff";
                statusWidget.value = "Ready to save prompts...";

                return result;
            };

            // Update widget when executed
            const onExecuted = nodeType.prototype.onExecuted;
            nodeType.prototype.onExecuted = function (message) {
                onExecuted?.apply(this, arguments);
                
                // Find the status display widget and update it
                const statusWidget = this.widgets.find(w => w.name === "status_display");
                if (statusWidget && message.text) {
                    // message.text is a tuple: (array_of_status, final_message)
                    const statusText = Array.isArray(message.text[0]) ? message.text[0].join('\n') : message.text[1];
                    statusWidget.value = statusText;
                }
            };
        }
    },
});