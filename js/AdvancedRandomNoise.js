import { app } from "../../scripts/app.js";

// Node extension for Advanced Random Noise
app.registerExtension({
    name: "AdvancedRandomNoise",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "AdvancedRandomNoise") {
            // Store original nodeCreated function
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            
            nodeType.prototype.onNodeCreated = function() {
                // Call original function if it exists
                if (onNodeCreated) {
                    onNodeCreated.apply(this, arguments);
                }
                
                // Add current seed display as a text display
                const currentSeedDisplay = this.addWidget("text", "current seed", "0", () => {}, {
                    readonly: true,
                    serialize: false // Don't serialize this in the workflow
                });
                
                // Style the display widget
                currentSeedDisplay.draw = function(ctx, node, widget_width, y, widget_height) {
                    // Draw background
                    ctx.fillStyle = "#2a2a2a";
                    ctx.fillRect(0, y, widget_width, widget_height);
                    
                    // Draw border
                    ctx.strokeStyle = "#555";
                    ctx.lineWidth = 1;
                    ctx.strokeRect(0, y, widget_width, widget_height);
                    
                    // Draw text
                    ctx.fillStyle = "#ccc";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "left";
                    const displayText = `current seed: ${Math.floor(this.value)}`;
                    ctx.fillText(displayText, 10, y + 17);
                };
                
                // Add copy button
                const copyButton = this.addWidget("button", "Use Current Seed", null, () => {
                    // Get current seed from the display widget
                    const currentSeed = Math.floor(currentSeedDisplay.value);
                    
                    // Find the noise_seed widget and set its value
                    const noiseSeedWidget = this.widgets.find(w => w.name === "noise_seed");
                    if (noiseSeedWidget) {
                        noiseSeedWidget.value = currentSeed;
                        console.log(`Advanced Random Noise: Copied seed ${currentSeed} to noise_seed input`);
                        
                        // Trigger update
                        if (this.onWidgetChanged) {
                            this.onWidgetChanged("noise_seed", currentSeed, currentSeed, noiseSeedWidget);
                        }
                    }
                });
                
                // Style the copy button
                copyButton.draw = function(ctx, node, widget_width, y, widget_height) {
                    // Draw button background
                    ctx.fillStyle = "#4a90e2";
                    ctx.fillRect(0, y, widget_width, widget_height);
                    
                    // Draw button border
                    ctx.strokeStyle = "#357abd";
                    ctx.lineWidth = 2;
                    ctx.strokeRect(0, y, widget_width, widget_height);
                    
                    // Draw button text
                    ctx.fillStyle = "#ffffff";
                    ctx.font = "12px Arial";
                    ctx.textAlign = "center";
                    ctx.fillText("Use Current Seed", widget_width / 2, y + 18);
                    ctx.textAlign = "left";
                };
                
                // Store references
                this.currentSeedDisplay = currentSeedDisplay;
                this.copyButton = copyButton;
                
                // Initialize current seed display with initial noise_seed value
                const noiseSeedWidget = this.widgets.find(w => w.name === "noise_seed");
                if (noiseSeedWidget) {
                    this.currentSeedDisplay.value = noiseSeedWidget.value;
                }
                
                console.log("Advanced Random Noise: Node UI initialized successfully");
            };
        }
    },
    
    setup() {
        // Listen to global queue events
        const origQueuePrompt = app.queuePrompt;
        app.queuePrompt = function(number, batchCount = 1) {
            // Before queuing, capture current seed values for all AdvancedRandomNoise nodes
            const graph = app.graph;
            if (graph && graph._nodes) {
                graph._nodes.forEach(node => {
                    if (node.type === "AdvancedRandomNoise") {
                        const noiseSeedWidget = node.widgets.find(w => w.name === "noise_seed");
                        if (noiseSeedWidget && node.currentSeedDisplay) {
                            node.currentSeedDisplay.value = noiseSeedWidget.value;
                            console.log(`Advanced Random Noise: Captured seed ${noiseSeedWidget.value} before queue`);
                            node.setDirtyCanvas(true);
                        }
                    }
                });
            }
            
            return origQueuePrompt.apply(this, arguments);
        };
    }
});

// Helper function to update seed display
function updateSeedDisplay(node, seed) {
    if (node.currentSeedDisplay) {
        node.currentSeedDisplay.value = seed;
        node.setDirtyCanvas(true);
    }
}

// Export for potential use by other modules
export { updateSeedDisplay };