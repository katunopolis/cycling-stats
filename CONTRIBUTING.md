# Contributing Guidelines

## Code Style

This project follows the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html). Key points:

### Documentation

1. Docstrings
   - Use triple quotes `"""`
   - First line should be a summary line
   - Leave one blank line after summary
   - Include Args:, Returns:, Raises: sections where applicable
   - Use imperative mood ("Do this", not "Does this")

2. Function Documentation
   ```python
   def function_name(param1: type1, param2: type2) -> return_type:
       """One-line summary of function purpose.

       Detailed description of function behavior.
       Multiple lines are allowed.

       Args:
           param1: Description of param1
           param2: Description of param2

       Returns:
           Description of return value

       Raises:
           ErrorType: Description of error conditions
       """
   ```

3. Module Documentation
   ```python
   """Module docstring appears at the top of the file.

   It should have a one-line summary followed by a blank line.
   Subsequent lines can provide detailed description.

   Typical usage example:
     foo = ClassFoo()
     bar = foo.method_bar()
   """
   ```

### Code Organization

1. Imports
   - Standard library imports first
   - Third-party imports second
   - Local imports third
   - Each group separated by a blank line

2. Global Variables
   - Use UPPER_CASE for constants
   - Include type hints where possible
   - Group related constants together
   - Add descriptive comments

3. Functions and Methods
   - Use snake_case for function names
   - Include type hints for parameters and return values
   - Keep functions focused and single-purpose
   - Maximum line length: 80 characters

### Error Handling

1. Use specific exception types
2. Include error messages that help diagnose the problem
3. Clean up resources in finally blocks
4. Document expected exceptions in docstrings

### Testing

1. Write unit tests for new functionality
2. Follow the Arrange-Act-Assert pattern
3. Use descriptive test names
4. Test edge cases and error conditions

## Version Control

1. Write clear commit messages
2. One logical change per commit
3. Reference issue numbers where applicable

## Pull Requests

1. Include a clear description of changes
2. Update documentation as needed
3. Add tests for new functionality
4. Ensure all tests pass
5. Follow existing code style 