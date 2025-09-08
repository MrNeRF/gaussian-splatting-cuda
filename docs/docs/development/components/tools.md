# Tools

1. **Create tool files**: `src/visualizer/tools/your_tool.hpp` and `.cpp`
   - Inherit from `ToolBase`, implement: `getName()`, `getDescription()`, `renderUI()`

2. **Register tool**: In `tool_manager.cpp` add to `registerBuiltinTools()`:
   ```cpp
   registry_.registerTool<YourTool>();
   ```

3. **Update build**: In `CMakeLists.txt` add:
   ```cmake
   tools/your_tool.cpp
   ```

4. **Build & run** - Your tool appears automatically in the Tools panel!

**Example minimal tool**: Copy `crop_box_tool.hpp/cpp`, rename class, change `getName()` to return your tool name.