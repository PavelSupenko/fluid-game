using System.Text;
using TMPro;
using UnityEngine;

namespace ThirdParty.MobileConsoleKit.Scripts.Utility
{
    public sealed class MemoryCounter : MonoBehaviour
    {
        [SerializeField] private TextMeshProUGUI _text;
        [SerializeField] private int _framesGate = 3;
        
        private readonly StringBuilder _stats = new();
        
        private void Update()
        {
            if (Time.frameCount % _framesGate != 0)
                return;
            
            _stats.Clear();
            _stats.AppendLine("TOTAL MEMORY");
            _stats.AppendLine($"Reserved: {ToMB(UnityEngine.Profiling.Profiler.GetTotalReservedMemoryLong())}");
            _stats.AppendLine($"Allocated: {ToMB(UnityEngine.Profiling.Profiler.GetTotalAllocatedMemoryLong())}");
            _stats.AppendLine($"Unused: {ToMB(UnityEngine.Profiling.Profiler.GetTotalUnusedReservedMemoryLong())}");

            _stats.AppendLine("\nMANAGED MEMORY");
            _stats.AppendLine($"Heap Size: {ToMB(UnityEngine.Profiling.Profiler.GetMonoHeapSizeLong())}");
            _stats.AppendLine($"Mono Used: {ToMB(UnityEngine.Profiling.Profiler.GetMonoUsedSizeLong())}");
            _stats.AppendLine($"GC Total: {ToMB(System.GC.GetTotalMemory(false))}");

            _stats.AppendLine("\nGRAPHICS & HARDWARE");
            _stats.AppendLine($"GPU Capacity: {SystemInfo.graphicsMemorySize}");
            _stats.AppendLine($"GPU Used: {ToMB(UnityEngine.Profiling.Profiler.GetAllocatedMemoryForGraphicsDriver())}");
            _stats.AppendLine($"RAM Capacity: {SystemInfo.systemMemorySize}");
            
            _text.text = _stats.ToString();
        }

        private string ToMB(long bytes) => (bytes / 1024f / 1024f).ToString("F2");
    }
}