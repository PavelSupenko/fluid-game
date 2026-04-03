using System.Collections.Generic;
using System.Globalization;
using System;
using System.Linq;
using Newtonsoft.Json;

namespace MeltIt.Services.Cheats
{
    public class IntCheatProperty : ACheatProperty<int>
    {
        public int? Min { get; }
        public int? Max { get; }

        public IntCheatProperty(int value, bool switchable = false, int? min = null, int? max = null) 
            : base(value, switchable)
        {
            Min = min;
            Max = max;
        }

        protected override bool TryParseValue(string? value, out int result) => 
            int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out result);
    }

    public class FloatCheatProperty : ACheatProperty<float>
    {
        public float? Min { get; }
        public float? Max { get; }
        
        public FloatCheatProperty(float value, bool switchable = false, float? min = null, float? max = null)
            : base(value, switchable)
        {
            Min = min;
            Max = max;
        }

        protected override bool TryParseValue(string? value, out float result) => 
            float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out result);
    }
    
    public class StringCheatProperty : ACheatProperty<string>
    {
        public StringCheatProperty(string value, bool switchable = false) : base(value, switchable) { }

        protected override bool TryParseValue(string? value, out string result)
        {
            result = value ?? string.Empty;
            return true; // Always succeeds for strings
        }
    }

    public class BoolCheatProperty : ACheatProperty<bool>
    {
        public BoolCheatProperty(bool value, bool switchable = false) : base(value, switchable) { }

        protected override bool TryParseValue(string? value, out bool result) => 
            bool.TryParse(value, out result);
    }

    public class EnumCheatProperty : ACheatProperty<string>
    {
        public Type EnumType { get; }
        
        public EnumCheatProperty(Type enumType, string value, bool switchable) : base(value, switchable) => 
            EnumType = enumType;

        protected override bool TryParseValue(string? value, out string result)
        {
            result = value ?? string.Empty;
            return true;
        }
    }

    public class ArrayCheatProperty : ACheatProperty<string>
    {
        public string[] Array { get; }
        
        public ArrayCheatProperty(string[] array, string value, bool switchable = false) : base(value, switchable) => 
            Array = array;

        protected override bool TryParseValue(string? value, out string result)
        {
            result = value ?? string.Empty;
            return true;
        }
    }
    
    public class TimeSpanCheatProperty : ACheatProperty<TimeSpan>
    {
        public TimeSpanCheatProperty(TimeSpan value, bool switchable = false) : base(value, switchable) { }
        
        public override string ToString() => 
            $"{(int)Value.TotalDays}d {Value.Hours}h {Value.Minutes}m";

        protected override bool TryParseValue(string? value, out TimeSpan result)
        {
            string[] parts = (value ?? string.Empty).Split(' ', StringSplitOptions.RemoveEmptyEntries);
            if (parts.Length == 0)
            {
                result = TimeSpan.Zero;
                return false;
            }
            
            string daysPart = parts.FirstOrDefault(p => p.EndsWith("d", StringComparison.OrdinalIgnoreCase)) ?? "0d";
            string hoursPart = parts.FirstOrDefault(p => p.EndsWith("h", StringComparison.OrdinalIgnoreCase)) ?? "0h";
            string minutesPart = parts.FirstOrDefault(p => p.EndsWith("m", StringComparison.OrdinalIgnoreCase)) ?? "0m";
            
            bool parsedDays = int.TryParse(daysPart[..^1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int days);
            bool parsedHours = int.TryParse(hoursPart[..^1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int hours);
            bool parsedMinutes = int.TryParse(minutesPart[..^1], NumberStyles.Integer, CultureInfo.InvariantCulture, out int minutes);
            
            result = new TimeSpan(days, hours, minutes, seconds: 0);
            return parsedDays || parsedHours || parsedMinutes;
        }
    }

    public interface ICheatProperty
    {
        event Action? ValueChanged;
        bool IsValid { get; }
        bool Switchable { get; }
        int BindingsCount { get; }
        void Activate();
        void Deactivate();
        void ResetFromStringValue(string? value);
        void UpdateValue();
        void Unbind(object sender);
        void UnbindAll();
    }
    
    public abstract class ACheatProperty<T> : ICheatProperty
    {
        public event Action? ValueChanged;
        
        [JsonIgnore] public bool IsValid => !_switchable || _overridden;
        [JsonIgnore] public bool Switchable => _switchable;

        [JsonIgnore] public T Value => _value;
        [JsonIgnore] public int BindingsCount => _bindings.Count;

        [JsonProperty] private bool _overridden;
        [JsonProperty] private T _value;

        private readonly Dictionary<object, Action<T>> _bindings = new();
        private readonly bool _switchable;
        private Func<T>? _getter;

        protected ACheatProperty(T value, bool switchable)
        {
            _value = value;
            _switchable = switchable;
            _overridden = false;
        }
        
        protected abstract bool TryParseValue(string? value, out T result);

        public void Activate()
        {
            _overridden = true;
            NotifyAboutChanges();
        }

        public void Deactivate()
        {
            _overridden = false;
            NotifyAboutChanges();
        }

        public void ResetFromStringValue(string? value)
        {
            if (TryParseValue(value, out T result))
            {
                _overridden = true;
                _value = result;
            }
            else
            {
                _overridden = false;
            }
            
            NotifyAboutChanges();
        }

        public void UpdateValue()
        {
            if (_getter != null) 
                _value = _getter();
        }

        public void Bind(object binder, Func<T> getter, Action<T> setter)
        {
            _getter = getter;
            if (TryGetValue(out T result))
                setter(result);
            else
                _value = getter();

            if (!_bindings.TryAdd(binder, setter)) 
                _bindings[binder] = setter;
        }

        public void Unbind(object binder)
        {
            _bindings.Remove(binder);
        }

        public void UnbindAll()
        {
            _bindings.Clear();
        }

        public bool TryGetValue(out T result)
        {
            result = _value;
            return IsValid;
        }

        private void NotifyAboutChanges()
        {
            if (TryGetValue(out T intValue))
            {
                foreach (var binding in _bindings.Values)
                    binding(intValue);
            }
            
            ValueChanged?.Invoke();
        }
    }
}