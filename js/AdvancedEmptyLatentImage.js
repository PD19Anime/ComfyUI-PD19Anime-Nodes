import { app } from "../../scripts/app.js";

app.registerExtension({
    name: "AdvancedEmptyLatentImage",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        console.log("Checking node:", nodeData.name);
        if (nodeData.name === "AdvancedEmptyLatentImage") {
            console.log("Registering AdvancedEmptyLatentImage extension");
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                console.log("AdvancedEmptyLatentImage node created, widgets:", this.widgets);

                // Function to update widget states
                const updateWidgetStates = () => {
                    console.log("Updating widget states");
                    const presetWidget = this.widgets.find(w => w.name === "preset");
                    const widthWidget = this.widgets.find(w => w.name === "width");
                    const heightWidget = this.widgets.find(w => w.name === "height");

                    console.log("Found widgets:", {
                        preset: presetWidget ? presetWidget.value : "not found",
                        width: widthWidget ? "found" : "not found",
                        height: heightWidget ? "found" : "not found"
                    });

                    if (presetWidget && widthWidget && heightWidget) {
                        const isManual = presetWidget.value === "Manual";
                        console.log("Is manual mode:", isManual);
                        
                        // Set the disabled property on the widgets directly
                        widthWidget.disabled = !isManual;
                        heightWidget.disabled = !isManual;
                        
                        console.log("Set width disabled to:", !isManual);
                        console.log("Set height disabled to:", !isManual);
                        
                        // Try to update visual appearance if DOM elements exist
                        const widthElement = widthWidget.element;
                        const heightElement = heightWidget.element;
                        
                        if (widthElement) {
                            if (isManual) {
                                widthElement.style.opacity = "1";
                                widthElement.style.backgroundColor = "";
                            } else {
                                widthElement.style.opacity = "0.6";
                                widthElement.style.backgroundColor = "#333";
                            }
                        }
                        
                        if (heightElement) {
                            if (isManual) {
                                heightElement.style.opacity = "1";
                                heightElement.style.backgroundColor = "";
                            } else {
                                heightElement.style.opacity = "0.6";
                                heightElement.style.backgroundColor = "#333";
                            }
                        }
                        
                        // Force the node to redraw to reflect the changes
                        this.setDirtyCanvas(true, true);
                    }
                };

                // Hook into preset widget value changes
                const presetWidget = this.widgets.find(w => w.name === "preset");
                if (presetWidget) {
                    console.log("Hooking into preset widget changes");
                    const originalCallback = presetWidget.callback;
                    presetWidget.callback = function() {
                        console.log("Preset changed to:", this.value);
                        if (originalCallback) {
                            originalCallback.apply(this, arguments);
                        }
                        updateWidgetStates();
                    };
                }

                // Initial state update
                setTimeout(updateWidgetStates, 500);

                return result;
            };

            // Also update when the node is configured (loaded from workflow)
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const result = onConfigure?.apply(this, arguments);
                
                // Update widget states after configuration
                setTimeout(() => {
                    const presetWidget = this.widgets.find(w => w.name === "preset");
                    const widthWidget = this.widgets.find(w => w.name === "width");
                    const heightWidget = this.widgets.find(w => w.name === "height");

                    if (presetWidget && widthWidget && heightWidget) {
                        const isManual = presetWidget.value === "Manual";
                        
                        widthWidget.disabled = !isManual;
                        heightWidget.disabled = !isManual;
                        
                        const widthElement = widthWidget.element;
                        const heightElement = heightWidget.element;
                        
                        if (widthElement) {
                            if (isManual) {
                                widthElement.style.opacity = "1";
                                widthElement.style.backgroundColor = "";
                            } else {
                                widthElement.style.opacity = "0.6";
                                widthElement.style.backgroundColor = "#333";
                            }
                        }
                        
                        if (heightElement) {
                            if (isManual) {
                                heightElement.style.opacity = "1";
                                heightElement.style.backgroundColor = "";
                            } else {
                                heightElement.style.opacity = "0.6";
                                heightElement.style.backgroundColor = "#333";
                            }
                        }
                    }
                }, 100);

                return result;
            };
        }
    },
});