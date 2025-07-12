import { app } from "../../scripts/app.js";
import { ComfyWidgets } from "../../scripts/widgets.js";
import { $el } from "../../scripts/ui.js";

const ADVANCED_MODEL_LOADER = "AdvancedModelLoader";

function getType(node, widgetName) {
    if (widgetName === "ckpt_name") {
        return "checkpoints";
    }
    return "loras";
}

function getWidgetName(type) {
    return type === "checkpoints" ? "ckpt_name" : "lora1_name";
}

app.registerExtension({
    name: "AdvancedModelLoader.TreeSelector",
    init() {
        $el("style", {
            textContent: `
                .pd19-combo-folder { 
                    opacity: 0.7; 
                    background-color: rgba(0, 0, 0, 0.1);
                }
                .pd19-combo-folder-arrow { 
                    display: inline-block; 
                    width: 15px; 
                    transition: transform 0.2s;
                }
                .pd19-combo-folder:hover { 
                    background-color: rgba(255, 255, 255, 0.1); 
                }
                .pd19-combo-prefix { 
                    display: none;
                    color: #888;
                }

                /* Special handling for when the filter input is populated */
                .litecontextmenu:has(input:not(:placeholder-shown)) .pd19-combo-folder-contents {
                    display: block !important;
                }
                .litecontextmenu:has(input:not(:placeholder-shown)) .pd19-combo-folder { 
                    display: none;
                }
                .litecontextmenu:has(input:not(:placeholder-shown)) .pd19-combo-prefix { 
                    display: inline;
                }
                .litecontextmenu:has(input:not(:placeholder-shown)) .litemenu-entry { 
                    padding-left: 2px !important;
                }
            `,
            parent: document.body,
        });

        const positionMenu = (menu) => {
            // Compute best position
            let left = app.canvas.last_mouse[0] - 10;
            let top = app.canvas.last_mouse[1] - 10;

            const body_rect = document.body.getBoundingClientRect();
            const root_rect = menu.getBoundingClientRect();

            if (body_rect.width && left > body_rect.width - root_rect.width - 10) {
                left = body_rect.width - root_rect.width - 10;
            }
            if (body_rect.height && top > body_rect.height - root_rect.height - 10) {
                top = body_rect.height - root_rect.height - 10;
            }

            menu.style.left = `${left}px`;
            menu.style.top = `${top}px`;
        };

        const createTree = (menu, items) => {
            // Create a map to store folder structures
            const folderMap = new Map();
            const rootItems = [];
            const splitBy = (navigator.platform || navigator.userAgent).includes("Win") ? /\/|\\/ : /\//;
            const itemsSymbol = Symbol("items");

            // First pass - organize items into folder structure
            for (const item of items) {
                const path = item.getAttribute("data-value").split(splitBy);
                
                // Remove path from visible text
                item.textContent = path[path.length - 1];
                if (path.length > 1) {
                    // Add the prefix path back in so it can be filtered on
                    const prefix = $el("span.pd19-combo-prefix", {
                        textContent: path.slice(0, -1).join("/") + "/",
                    });
                    item.prepend(prefix);
                }

                if (path.length === 1) {
                    rootItems.push(item);
                    continue;
                }

                // Temporarily remove the item from current position
                item.remove();

                // Create folder hierarchy
                let currentLevel = folderMap;
                for (let i = 0; i < path.length - 1; i++) {
                    const folder = path[i];
                    if (!currentLevel.has(folder)) {
                        currentLevel.set(folder, new Map());
                    }
                    currentLevel = currentLevel.get(folder);
                }

                // Store the actual item in the deepest folder
                if (!currentLevel.has(itemsSymbol)) {
                    currentLevel.set(itemsSymbol, []);
                }
                currentLevel.get(itemsSymbol).push(item);
            }

            const createFolderElement = (name) => {
                const folder = $el("div.litemenu-entry.pd19-combo-folder", {
                    innerHTML: `<span class="pd19-combo-folder-arrow">▶</span> ${name}`,
                    style: { paddingLeft: "5px" },
                });
                return folder;
            };

            const insertFolderStructure = (parentElement, map, level = 0) => {
                for (const [folderName, content] of map.entries()) {
                    if (folderName === itemsSymbol) continue;

                    const folderElement = createFolderElement(folderName);
                    folderElement.style.paddingLeft = `${level * 15 + 5}px`;
                    parentElement.appendChild(folderElement);

                    const childContainer = $el("div.pd19-combo-folder-contents", {
                        style: { display: "none" },
                    });

                    // First, recursively add subfolders (they will appear at the top)
                    insertFolderStructure(childContainer, content, level + 1);

                    // Then add items in this folder (they will appear after subfolders)
                    const items = content.get(itemsSymbol) || [];
                    for (const item of items) {
                        item.style.paddingLeft = `${(level + 1) * 15 + 14}px`;
                        childContainer.appendChild(item);
                    }
                    parentElement.appendChild(childContainer);

                    // Add click handler for folder
                    folderElement.addEventListener("click", (e) => {
                        e.stopPropagation();
                        const arrow = folderElement.querySelector(".pd19-combo-folder-arrow");
                        const contents = folderElement.nextElementSibling;
                        if (contents.style.display === "none") {
                            contents.style.display = "block";
                            arrow.textContent = "▼";
                            arrow.style.transform = "rotate(90deg)";
                        } else {
                            contents.style.display = "none";
                            arrow.textContent = "▶";
                            arrow.style.transform = "rotate(0deg)";
                        }
                    });
                }
            };

            // Insert root items first
            for (const item of rootItems) {
                item.style.paddingLeft = "5px";
            }

            // Insert folder structure
            insertFolderStructure(items[0]?.parentElement || menu, folderMap);
            positionMenu(menu);
        };

        const updateMenu = (menu, node, widgetName) => {
            // Clamp max height so it doesn't overflow the screen
            const position = menu.getBoundingClientRect();
            const maxHeight = window.innerHeight - position.top - 20;
            menu.style.maxHeight = `${maxHeight}px`;

            const items = menu.querySelectorAll(".litemenu-entry");
            
            if (items.length > 0) {
                createTree(menu, items);
            }
        };

        const mutationObserver = new MutationObserver((mutations) => {
            const node = app.canvas.current_node;

            if (!node || node.comfyClass !== ADVANCED_MODEL_LOADER) {
                return;
            }

            for (const mutation of mutations) {
                for (const added of mutation.addedNodes) {
                    if (added.classList?.contains("litecontextmenu")) {
                        const overWidget = app.canvas.getWidgetAtCursor();
                        
                        if (overWidget?.name === "ckpt_name" || 
                            overWidget?.name === "lora1_name" || 
                            overWidget?.name === "lora2_name") {
                            
                            requestAnimationFrame(() => {
                                // Check for the filter input to ensure it's a combo dropdown
                                if (!added.querySelector(".comfy-context-menu-filter")) return;
                                
                                updateMenu(added, node, overWidget.name);
                            });
                        }
                        return;
                    }
                }
            }
        });

        mutationObserver.observe(document.body, { childList: true, subtree: false });
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === ADVANCED_MODEL_LOADER) {
            console.log("Registering AdvancedModelLoader tree selector extension");
            
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const result = onNodeCreated?.apply(this, arguments);
                console.log("AdvancedModelLoader node created, widgets:", this.widgets);

                // Function to update widget states based on enable/disable switches
                const updateWidgetStates = () => {
                    const lora1EnabledWidget = this.widgets.find(w => w.name === "lora1");
                    const lora2EnabledWidget = this.widgets.find(w => w.name === "lora2");
                    const lora1Widget = this.widgets.find(w => w.name === "lora1_name");
                    const lora1StrengthWidget = this.widgets.find(w => w.name === "lora1_strength");
                    const lora2Widget = this.widgets.find(w => w.name === "lora2_name");
                    const lora2StrengthWidget = this.widgets.find(w => w.name === "lora2_strength");

                    if (lora1EnabledWidget && lora1Widget && lora1StrengthWidget) {
                        const lora1Enabled = lora1EnabledWidget.value;
                        lora1Widget.disabled = !lora1Enabled;
                        lora1StrengthWidget.disabled = !lora1Enabled;
                        
                        // Update visual appearance
                        if (lora1Widget.element) {
                            lora1Widget.element.style.opacity = lora1Enabled ? "1" : "0.5";
                        }
                        if (lora1StrengthWidget.element) {
                            lora1StrengthWidget.element.style.opacity = lora1Enabled ? "1" : "0.5";
                        }
                    }

                    if (lora2EnabledWidget && lora2Widget && lora2StrengthWidget) {
                        const lora2Enabled = lora2EnabledWidget.value;
                        lora2Widget.disabled = !lora2Enabled;
                        lora2StrengthWidget.disabled = !lora2Enabled;
                        
                        // Update visual appearance
                        if (lora2Widget.element) {
                            lora2Widget.element.style.opacity = lora2Enabled ? "1" : "0.5";
                        }
                        if (lora2StrengthWidget.element) {
                            lora2StrengthWidget.element.style.opacity = lora2Enabled ? "1" : "0.5";
                        }
                    }
                };

                // Hook into enable/disable widget changes
                const lora1EnabledWidget = this.widgets.find(w => w.name === "lora1");
                const lora2EnabledWidget = this.widgets.find(w => w.name === "lora2");

                if (lora1EnabledWidget) {
                    const originalCallback1 = lora1EnabledWidget.callback;
                    lora1EnabledWidget.callback = function() {
                        console.log("LoRA1 enabled changed to:", this.value);
                        if (originalCallback1) {
                            originalCallback1.apply(this, arguments);
                        }
                        updateWidgetStates();
                    };
                }

                if (lora2EnabledWidget) {
                    const originalCallback2 = lora2EnabledWidget.callback;
                    lora2EnabledWidget.callback = function() {
                        console.log("LoRA2 enabled changed to:", this.value);
                        if (originalCallback2) {
                            originalCallback2.apply(this, arguments);
                        }
                        updateWidgetStates();
                    };
                }

                // Initial state update
                setTimeout(updateWidgetStates, 100);

                return result;
            };

            // Also update when the node is configured (loaded from workflow)
            const onConfigure = nodeType.prototype.onConfigure;
            nodeType.prototype.onConfigure = function () {
                const result = onConfigure?.apply(this, arguments);
                
                // Update widget states after configuration
                setTimeout(() => {
                    const lora1EnabledWidget = this.widgets.find(w => w.name === "lora1");
                    const lora2EnabledWidget = this.widgets.find(w => w.name === "lora2");
                    const lora1Widget = this.widgets.find(w => w.name === "lora1_name");
                    const lora1StrengthWidget = this.widgets.find(w => w.name === "lora1_strength");
                    const lora2Widget = this.widgets.find(w => w.name === "lora2_name");
                    const lora2StrengthWidget = this.widgets.find(w => w.name === "lora2_strength");

                    if (lora1EnabledWidget && lora1Widget && lora1StrengthWidget) {
                        const lora1Enabled = lora1EnabledWidget.value;
                        lora1Widget.disabled = !lora1Enabled;
                        lora1StrengthWidget.disabled = !lora1Enabled;
                        
                        if (lora1Widget.element) {
                            lora1Widget.element.style.opacity = lora1Enabled ? "1" : "0.5";
                        }
                        if (lora1StrengthWidget.element) {
                            lora1StrengthWidget.element.style.opacity = lora1Enabled ? "1" : "0.5";
                        }
                    }

                    if (lora2EnabledWidget && lora2Widget && lora2StrengthWidget) {
                        const lora2Enabled = lora2EnabledWidget.value;
                        lora2Widget.disabled = !lora2Enabled;
                        lora2StrengthWidget.disabled = !lora2Enabled;
                        
                        if (lora2Widget.element) {
                            lora2Widget.element.style.opacity = lora2Enabled ? "1" : "0.5";
                        }
                        if (lora2StrengthWidget.element) {
                            lora2StrengthWidget.element.style.opacity = lora2Enabled ? "1" : "0.5";
                        }
                    }
                }, 100);

                return result;
            };
        }
    },
});