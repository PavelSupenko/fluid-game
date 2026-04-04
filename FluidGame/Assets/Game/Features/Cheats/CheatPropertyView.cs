#if DebugLog
using MeltIt.Services.Cheats;
using MobileConsole.UI;
using UnityEngine;
using System;
using System.Globalization;

namespace MeltIt.Features.Cheats
{
    public class CheatPropertyView
    {
        private readonly ICheatProperty _property;
        private readonly ViewBuilder _view;
        private readonly Node _parent;
        private readonly string _name;

        private CheckboxNodeView _overrideOption;
        private NodeView _inputNode;

        public NodeView MainNode => _overrideOption ?? _inputNode;
        public ICheatProperty Property => _property;

        public CheatPropertyView(string name, ICheatProperty property, ViewBuilder view, Node parent)
        {
            _property = property;
            _parent = parent;
            _view = view;
            _name = name;
            Create();
        }

        public void UpdateProperty()
        {
            _property.UpdateValue();
        }

        public void UpdateView()
        {
            UpdateInputNode();
        }

        private void Create()
        {
            if (_property.Switchable)
            {
                _overrideOption = _view.AddCheckbox(
                    $"Override {_name}", _property.IsValid, OnOverrideOptionChanged, _parent);

                if (_property.IsValid)
                    OnOverrideOptionChanged(_overrideOption);
            }
            else
            {
                CreateInputNode(_parent);
            }
        }

        private void OnOverrideOptionChanged(CheckboxNodeView overrideToggle)
        {
            if (overrideToggle.isOn)
            {
                CreateInputNode(overrideToggle);
                _property.Activate();
            }
            else
            {
                RemoveInputNode();
                _property.Deactivate();
            }
            
            _view.Rebuild();
        }

        private void CreateInputNode(Node parent)
        {
            if (_inputNode == null)
            {
                switch (_property)
                {
                    case IntCheatProperty intProperty:
                        _inputNode = intProperty.Min != null && intProperty.Max != null
                            ? _view.AddSlider(_name, intProperty.Value, intProperty.Min.Value, intProperty.Max.Value,
                                true, OnInputValueChanged, string.Empty, null)
                            : _view.AddInput(_name, intProperty.Value.ToString(), true, OnInputValueChanged,
                                string.Empty, null);
                        break;
                    case FloatCheatProperty floatProperty: 
                        _inputNode = floatProperty.Min != null && floatProperty.Max != null
                            ? _view.AddSlider(_name, floatProperty.Value, floatProperty.Min.Value, floatProperty.Max.Value,
                                false, OnInputValueChanged, string.Empty, null)
                            : _view.AddInput(_name, floatProperty.Value.ToString(CultureInfo.InvariantCulture), true,
                                OnInputValueChanged, string.Empty, null);
                        break;
                    case StringCheatProperty stringProperty:
                        _inputNode = _view.AddInput(_name, stringProperty.Value, false, OnInputValueChanged, string.Empty, null);
                        break;
                    case BoolCheatProperty boolProperty:
                        _inputNode = _view.AddCheckbox(_name, boolProperty.Value, OnInputValueChanged, null);
                        break;
                    case EnumCheatProperty enumProperty:
                        string[] enumOptions = Enum.GetNames(enumProperty.EnumType);
                        int enumValueIndex = Array.IndexOf(enumOptions, enumProperty.Value);
                        _inputNode = _view.AddDropdown(_name, enumValueIndex, enumOptions, OnInputValueChanged, null);
                        break;
                    case ArrayCheatProperty arrayCheatProperty:
                        string[] arrayOptions = arrayCheatProperty.Array;
                        int arrayValueIndex = Array.IndexOf(arrayOptions, arrayCheatProperty.Value);
                        _inputNode = _view.AddDropdown(_name, arrayValueIndex, arrayOptions, OnInputValueChanged, null);
                        break;
                    case TimeSpanCheatProperty timeSpanCheatProperty:
                        TimeSpan timeSpan = timeSpanCheatProperty.Value;
                        string currentValue = $"{(int)timeSpan.TotalDays}d {timeSpan.Hours}h {timeSpan.Minutes}m";
                        _inputNode = _view.AddInput($"{_name} (35d 2h 1m)", currentValue, false, OnInputValueChanged, string.Empty, null);
                        break;
                    default:
                        throw new ArgumentOutOfRangeException();
                }
            }
            
            _inputNode.RemoveFromParent();
            parent.AddNode(_inputNode);
        }
        
        private void UpdateInputNode()
        {
            if (_inputNode == null)
                return;

            switch (_property)
            {
                case IntCheatProperty intProperty:
                    if (_inputNode is SliderNodeView slider)
                        slider.value = intProperty.Value;
                    else if (_inputNode is InputNodeView input) 
                        input.value = intProperty.Value.ToString();
                    break;
                case FloatCheatProperty floatProperty:
                    if (_inputNode is SliderNodeView floatSlider)
                        floatSlider.value = floatProperty.Value;
                    else if (_inputNode is InputNodeView floatInput) 
                        floatInput.value = floatProperty.Value.ToString(CultureInfo.InvariantCulture);
                    break;
                case StringCheatProperty stringProperty:
                    if (_inputNode is InputNodeView stringInput) 
                        stringInput.value = stringProperty.Value;
                    break;
                case BoolCheatProperty boolProperty:
                    if (_inputNode is CheckboxNodeView checkbox) 
                        checkbox.isOn = boolProperty.Value;
                    break;
                case EnumCheatProperty enumProperty:
                    if (_inputNode is DropdownNodeView dropdown) 
                        dropdown.index = Array.IndexOf(Enum.GetNames(enumProperty.EnumType), enumProperty.Value);
                    break;
                case ArrayCheatProperty arrayCheatProperty:
                    if (_inputNode is DropdownNodeView arrayDropdown) 
                        arrayDropdown.index = Array.IndexOf(arrayCheatProperty.Array, arrayCheatProperty.Value);
                    break;
                case TimeSpanCheatProperty timeSpanCheatProperty:
                    if (_inputNode is InputNodeView timeSpanInput) 
                        timeSpanInput.value = timeSpanCheatProperty.ToString();
                    break;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }

        private void RemoveInputNode()
        {
            _inputNode.RemoveFromParent();
        }

        private void OnInputValueChanged(DropdownNodeView inputField, int index)
        {
            try { _property.ResetFromStringValue(inputField.options[index]); }
            catch (Exception ex) { Debug.LogError($"Error parsing value for {_name}: {ex.Message}"); }
        }
        
        private void OnInputValueChanged(InputNodeView inputField)
        {
            try { _property.ResetFromStringValue(inputField.value); }
            catch (Exception ex) { Debug.LogError($"Error parsing value for {_name}: {ex.Message}"); }
        }

        private void OnInputValueChanged(CheckboxNodeView inputField)
        {
            try { _property.ResetFromStringValue(inputField.isOn.ToString()); }
            catch (Exception ex) { Debug.LogError($"Error parsing value for {_name}: {ex.Message}"); }
        }
        
        private void OnInputValueChanged(SliderNodeView inputField, float value)
        {
            try { _property.ResetFromStringValue(value.ToString(CultureInfo.InvariantCulture)); }
            catch (Exception ex) { Debug.LogError($"Error parsing value for {_name}: {ex.Message}"); }
        }
    }
}
#endif